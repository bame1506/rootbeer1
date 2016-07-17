package org.trifort.rootbeer.runtime;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.runtime.util.Stopwatch;
import org.trifort.rootbeer.runtimegpu.GpuException;
import org.trifort.rootbeer.util.ResourceReader;

/**
 * https://github.com/LMAX-Exchange/disruptor/wiki/Introduction
 * Library for passing event-messages to another thread which implements an
 * EventHandler (?) @see GpuEventHandler.
 */
import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.dsl.Disruptor;

import org.trifort.rootbeer.generate.bytecode.Constants;

public class CUDAContext implements Context
{

    final private GpuDevice              m_gpuDevice           ;
    final private boolean                m_is32bit             ;

    private long                         m_nativeContext       ;
    private long                         m_memorySize          ; /**<- bytes */
    private byte[]                       m_cubinFile           ;
    private Memory                       m_objectMemory        ;
    private Memory                       m_handlesMemory       ;
    private Memory                       m_textureMemory       ;
    private Memory                       m_exceptionsMemory    ;
    private Memory                       m_classMemory         ;
    private boolean                      m_usingUncheckedMemory; /**<- don't add out of bound heap debug output */
    private long                         m_requiredMemorySize  ;
    private CacheConfig                  m_cacheConfig         ;
    private ThreadConfig                 m_threadConfig        ;
    private Kernel                       m_kernelTemplate      ; /**<- ??? difference between this and compiledKernel? */
    private CompiledKernel               m_compiledKernel      ;
    private boolean                      m_usingHandles        ;

    /* Performance metrics / debug variables */
    final private StatsRow               m_stats               ;
    final private Stopwatch              m_readBlocksStopwatch ; /**<- device->host memcpy timer */

    final private ExecutorService        m_exec                ;
    final private Disruptor   <GpuEvent> m_disruptor           ;
    final private EventHandler<GpuEvent> m_handler             ;
    final private RingBuffer  <GpuEvent> m_ringBuffer          ;

    /* This is a static constructor for initializing all static members once.
     * Unlike the constructor this function is only called once on program
     * start (or first use of this class) and not on each new object creation.
     * This also means when initializing multiple GPU devices this function
     * is only called once. */
    static {
        initializeDriver();
    }

    public CUDAContext( final GpuDevice device )
    {
        /*** start up parallel event handling system ***/
        /* exec is an anonymous class which extends ThreadFactory by the
         * method newThread */
        m_exec = Executors.newCachedThreadPool( new ThreadFactory() {
            /* overrides same function in ThreadFactory, which is why it can't
             * be declared static, even though we don't access members */
            public Thread newThread( Runnable r )
            {
                final Thread t = new Thread(r);
                t.setDaemon( true );
                return t;
            }
        } );
        m_disruptor            = new Disruptor<GpuEvent>( GpuEvent.EVENT_FACTORY, 64, m_exec );
        m_handler              = new GpuEventHandler();
        m_disruptor.handleEventsWith( m_handler );
        m_ringBuffer           = m_disruptor.start();
        /* the started event handler will exist as long as the object to this
         * class */

        m_gpuDevice            = device;
        m_memorySize           = -1;    /* automatically determine size */

        final String arch      = System.getProperty("os.arch");
        m_is32bit              = arch.equals("x86") || arch.equals("i386");

        m_usingUncheckedMemory = true;
        m_usingHandles         = false;
        m_nativeContext        = allocateNativeContext();
        m_cacheConfig          = CacheConfig.PREFER_NONE;

        m_stats                = new StatsRow();
        m_readBlocksStopwatch  = new Stopwatch();
    }

    @Override
    public void close()
    {
        m_disruptor.shutdown();
        m_exec.shutdown();
        freeNativeContext( m_nativeContext );
        if ( m_objectMemory     != null ) m_objectMemory    .close();
        if ( m_handlesMemory    != null ) m_handlesMemory   .close();
        if ( m_exceptionsMemory != null ) m_exceptionsMemory.close();
        if ( m_classMemory      != null ) m_classMemory     .close();
        if ( m_textureMemory    != null ) m_textureMemory   .close();
    }

    /* Accessor methods boilerplate code */
    /* returns the GPU Device by wich this context was created */
    @Override public GpuDevice getDevice   () { return m_gpuDevice;          }
    @Override public long getRequiredMemory() { return m_requiredMemorySize; }
    @Override public StatsRow getStats     () { return m_stats;              }
    @Override public void setMemorySize  ( long         memorySize   ){ m_memorySize           = memorySize;   }
    @Override public void setCacheConfig ( CacheConfig  cacheConfig  ){ m_cacheConfig          = cacheConfig;  }
    @Override public void setThreadConfig( ThreadConfig threadConfig ){ m_threadConfig         = threadConfig; }
    @Override public void setUsingHandles( boolean      value        ){ m_usingHandles         = value;        }
    @Override public void useCheckedMemory()                          { m_usingUncheckedMemory = false;        }

    @Override
    public void setKernel( final Kernel kernelTemplate )
    {
        m_kernelTemplate = kernelTemplate;
        m_compiledKernel = (CompiledKernel) kernelTemplate;
    }

    /* Just a method which emulates default arguments */
    @Override
    public void setThreadConfig
    (
        final int threadCountX,
        final int blockCountX ,
        final int numThreads
    )
    {
        setThreadConfig( threadCountX, 1, 1, blockCountX, 1, numThreads );
    }
    /* Just a method which emulates default arguments */
    @Override
    public void setThreadConfig
    (
        final int threadCountX,
        final int threadCountY,
        final int blockCountX ,
        final int blockCountY ,
        final int numThreads
    )
    {
        setThreadConfig( threadCountX, threadCountY, 1, blockCountX, blockCountY, numThreads );
    }
    /* simple accessor function for setting the associated thread configuration */
    @Override
    public void setThreadConfig
    (
        final int threadCountX,
        final int threadCountY,
        final int threadCountZ,
        final int blockCountX ,
        final int blockCountY ,
        final int numThreads
    )
    {
        m_threadConfig = new ThreadConfig(
            threadCountX, threadCountY, threadCountZ,
            blockCountX , blockCountY , numThreads
        );
    }

    /* Seems to load cubin file and allocates memory for member 'compiledKernel',
     * therefore 'setKernel' must be called prior to this! */
    @Override
    public void buildState()
    {
        String  filename;
        int     size  = 0;
        boolean error = false;

        if ( m_is32bit )
        {
            filename = m_compiledKernel.getCubin32();
            size     = m_compiledKernel.getCubin32Size();
            error    = m_compiledKernel.getCubin32Error();
        }
        else
        {
            filename = m_compiledKernel.getCubin64();
            size     = m_compiledKernel.getCubin64Size();
            error    = m_compiledKernel.getCubin64Error();
        }

        if ( error ) {
            throw new RuntimeException("CUDA code compiled with error");
        }

        m_cubinFile = readCubinFile( filename, size );

        if ( m_usingUncheckedMemory )
        {
            m_classMemory        = new FixedMemory(1024); /* why 1024 ??? */
            m_exceptionsMemory   = new FixedMemory(getExceptionsMemSize(m_threadConfig));
            m_textureMemory      = new FixedMemory(8);
            if ( m_usingHandles ) {
                m_handlesMemory  = new FixedMemory(4*m_threadConfig.getNumThreads());
            } else {
                m_handlesMemory  = new FixedMemory(4);
            }
        }
        else
        {
            /* exactly same as FixedMemory block above */
            m_classMemory        = new CheckedFixedMemory(1024);
            m_exceptionsMemory   = new CheckedFixedMemory(getExceptionsMemSize(m_threadConfig));
            m_textureMemory      = new CheckedFixedMemory(8);
            if ( m_usingHandles ) {
                m_handlesMemory  = new CheckedFixedMemory(4*m_threadConfig.getNumThreads());
            } else {
                m_handlesMemory  = new CheckedFixedMemory(4);
            }
        }
        /* findMemory size needs m_classMemory, m_textureMemory to be set! */
        if ( m_memorySize == -1 ) {
            findMemorySize( );
        }
        if ( m_usingUncheckedMemory ) {
            m_objectMemory = new FixedMemory( m_memorySize );
        }   else {
            m_objectMemory = new CheckedFixedMemory( m_memorySize );
        }

        final long seq = m_ringBuffer.next();
        GpuEvent gpuEvent = m_ringBuffer.get( seq );
        gpuEvent.setValue( GpuEventCommand.NATIVE_BUILD_STATE );
        gpuEvent.getFuture().reset();
        m_ringBuffer.publish( seq );
        gpuEvent.getFuture().take(); /* wait for NATIVE_BUILD_STATE to have finished */
    }

    private long getExceptionsMemSize( ThreadConfig thread_config )
    {
        /* 4L seems a bit magic number to me, it seems to be sizeof( Exception ) */
        if ( Configuration.runtimeInstance().getExceptions() ) {
            return 4L * thread_config.getNumThreads();
        } else {
            return 4;
        }
    }

    /**
     * This wrapper catches IOException as forced to do so by compiler:
     * error: unreported exception IOException; must be caught or declared to be thrown
     */
    private byte[] readCubinFile( String filename, int length )
    {
        try
        {
            byte[] buffer = ResourceReader.getResourceArray(filename, length);
            return buffer;
        }
        catch ( Exception ex )
        {
	       ex.printStackTrace();
           throw new RuntimeException(ex);
        }
    }

    /**
     * Automatically finds a good memory size needed for allocation from
     * several parameters. Also checks if the free memory is enough to hold it.
     * @return sets m_memorySize to needed size in bytes for kernel serialization
     */
    private void findMemorySize()
    {
        final long freeMemSizeGPU = m_gpuDevice.getFreeGlobalMemoryBytes();
        final long freeMemSizeCPU = Runtime.getRuntime().freeMemory(); /* bytes */

        //objectMemory  = new FixedMemoryDummy( 1024 );
        //textureMemory = new FixedMemoryDummy( 1024 );
        //final Serializer serial = compiledKernel.getSerializer( objectMemory, textureMemory );
        //System.out.println( " doGetSize = "+serial.doGetSize( compiledKernel ) );
        //serial.writeToHeap( compiledKernel );
        //System.out.println( " doGetSize = "+serial.doGetSize( compiledKernel ) );
        //System.out.println( " objectMemory.getSize = "+serial.doGetSize( compiledKernel ) );
        /* in the worst case classMemory is only bytes which all would get
         * aligned to 16-byte boundarys (Constants.MallocAlignBytes) resulting
         * in 16-fold memory needed. Exception size are assumed to be 4 bytes
         * and also assumed to be aligned (are they???) */
        /* absolutely fucking wrong everything in here ... */
        final long neededMemory =
            m_cubinFile.length + Constants.MallocAlignBytes +
            m_exceptionsMemory.getSize() / 4 * Constants.MallocAlignBytes +
            m_classMemory.getSize() /* * Constants.MallocAlignBytes */ * m_threadConfig.getNumThreads() +
            m_handlesMemory.getSize() + 1024*1024 /* 1 MB buffer... this really needs some correct formula or better take -.- */;
        // this is the external formula I cam up with:
        //     createContext(
        //     ( work.size * 2 /* nIteration and nHits List */ * 8 /* sizeof(Long) */ +
        //       work.size * 4 /* sizeof(exception) ??? */ ) * 4 /* empirical factor */ +
        //       2*1024*1024 /* safety padding */
        // )
        /* After some bisection on a node with two K80 GPUs (26624 max. threads)
         * I found 2129920 B to be too few and 2129920+1024 sufficient.
         * The later means
         *     safety padding is : 1598464 B
         *      calculated size  :  532480 B = max.Threads * ( 2 * 8 + 4 )
         * The total size needed is therefore almost exactly 4 times the size
         * calculated!
         * Further tests show that max.Threads * ( 2 * 8 + 4 ) * 4 + pad
         * works for pad = 192, but not for 191
         * K20x (28672 max. threads) failed with the pad of 192
         * (total size: 2293952 B) , a pad of 1024*1024 worked, though.
         **/



        final String debugOutput =
            "  Debugging Output:\n"                                              +
            "    GPU size         : " + freeMemSizeGPU                    + " B\n" +
            "    CPU_SIZE         : " + freeMemSizeCPU                    + " B\n" +
            "    Exceptions size  : " + m_exceptionsMemory.getSize()      + " B\n" +
            "    class memory size: " + m_classMemory.getSize()           + " B\n" +
            "    cubin size       : " + m_cubinFile.length                + " B\n" +
            "    cubin32 size     : " + m_compiledKernel.getCubin32Size() + " B\n" +
            "    cubin64 size     : " + m_compiledKernel.getCubin64Size() + " B\n" +
            "    alignment        : " + Constants.MallocAlignBytes        + " B\n" ;
        System.out.print( debugOutput );
        if ( neededMemory > Math.min( freeMemSizeGPU, freeMemSizeCPU ) ) {
            final String error =
                "OutOfMemory while allocating Java CPU and GPU memory.\n"     +
                "  Try increasing the max Java Heap Size using -Xmx and the " +
                "  initial Java Heap Size using -Xms.\n"                      +
                "  Try reducing the number of threads you are using.\n"       +
                "  Try using kernel templates.\n"                             ;
            throw new RuntimeException( error + debugOutput );
        }
        m_memorySize = neededMemory;
    }

  @Override
  public void run()
  {
      GpuFuture future = runAsync();
      future.take();
  }

  /**
   * @todo Is this and run without a list actually being used?
   * Normally Rootbeer.java:run will be used and that only works with a
   * list of kernels
   */
  @Override
  public GpuFuture runAsync()
  {
      final long seq = m_ringBuffer.next();
      GpuEvent gpuEvent = m_ringBuffer.get(seq);
          gpuEvent.setValue(GpuEventCommand.NATIVE_RUN);
      gpuEvent.getFuture().reset();
      m_ringBuffer.publish(seq);
      return gpuEvent.getFuture();
  }

    /**
     * Launches a kernel asynchronously
     */
    @Override
    public GpuFuture runAsync( final List<Kernel> work )
    {
        final long seq = m_ringBuffer.next();
        GpuEvent gpuEvent = m_ringBuffer.get( seq );
            gpuEvent.setKernelList( work );
            gpuEvent.setValue(GpuEventCommand.NATIVE_RUN_LIST);
        gpuEvent.getFuture().reset();
        m_ringBuffer.publish(seq);
        return gpuEvent.getFuture();
    }

    @Override
    public void run( final List<Kernel> work )
    {
        GpuFuture future = runAsync(work);
        future.take();
    }

    /**
     * Implements onEvent with NATIVE_BUILD_STATE NATIVE_RUN and NATIVE_RUN_LIST
     * Those are all wrappers to JNI functions, for documentation
     * @see CudaContext.c
     *    Java_org_trifort_rootbeer_runtime_CUDAContext_cudaRun
     *    Java_org_trifort_rootbeer_runtime_CUDAContext_nativeBuildState
     **/
    private class GpuEventHandler implements EventHandler<GpuEvent>
    {
        @Override
        public void onEvent
        (
            final GpuEvent gpuEvent,
            final long     sequence,
            final boolean  endOfBatch
        )
        {
            try {
                switch ( gpuEvent.getValue() )
                {
                    case NATIVE_BUILD_STATE:
                    {
                        final boolean usingExceptions = Configuration.runtimeInstance().getExceptions();
                        nativeBuildState( m_nativeContext, m_gpuDevice.getDeviceId(), m_cubinFile,
                            m_cubinFile.length,
                            m_threadConfig.getThreadCountX(),
                            m_threadConfig.getThreadCountY(),
                            m_threadConfig.getThreadCountZ(),
                            m_threadConfig.getBlockCountX (),
                            m_threadConfig.getBlockCountY (),
                            m_threadConfig.getNumThreads  (),
                            m_objectMemory                  ,
                            m_handlesMemory                 ,
                            m_exceptionsMemory              ,
                            m_classMemory                   ,
                            usingExceptions ? 1 : 0         ,
                            m_cacheConfig.ordinal()
                        );
                        gpuEvent.getFuture().signal();
                        break;
                    }
                    case NATIVE_RUN:
                    {
                        /* send Kernel members to GPU (serializing) */
                        writeBlocksTemplate();
                        runGpu();
                        /* get possibly changed Kernel members back from GPU */
                        readBlocksTemplate();
                        gpuEvent.getFuture().signal();
                        break;
                    }
                    case NATIVE_RUN_LIST:
                    {
                        writeBlocksList( gpuEvent.getKernelList() );
                        runGpu();
                        readBlocksList(  gpuEvent.getKernelList() );
                        gpuEvent.getFuture().signal();
                        break;
                    }
                    default:
                    {
                        throw new RuntimeException(
                            "[CUDAContext.java] Unknown GPU event command code : " +
                            gpuEvent.getValue()
                        );
                    }
                }
            }
            catch ( Exception ex )
            {
                gpuEvent.getFuture().setException(ex);
                gpuEvent.getFuture().signal();
            }
        }
    }

    /* @see writeBlocksList(List<Kernel> work) */
    private void writeBlocksTemplate()
    {
        /* function body could be replaced with this:
         *     handlesMemory.setAddress(0);
         *     writeBlocksTemplate( List<Kernel>( compiledKernel ) );
         *   => this is not easily possible, because writeBlocksList takes
         *      a list of Kernel not CompiledKernel
         */
        final Stopwatch watch = new Stopwatch();
        watch.start();
        m_objectMemory.clearHeapEndPtr();
        m_handlesMemory.setAddress(0);

        final Serializer serializer = m_compiledKernel.getSerializer(m_objectMemory, m_textureMemory);
        serializer.writeStaticsToHeap();

        final long handle = serializer.writeToHeap( m_compiledKernel );
        m_handlesMemory.writeRef(handle);
        m_objectMemory.align16();

        if ( Configuration.getPrintMem() )
        {
            final BufferPrinter printer = new BufferPrinter();
            printer.print( m_objectMemory, 0, 256 );
        }

        watch.stop();
        m_stats.setSerializationTime( watch.elapsedTimeMillis() );
    }

    /**
     * This seems to serialize and send to the GPU all the Kernel objects.
     *
     * This is used by GpuEvent NATIVE_RUN[_LIST] which is only used by
     * runAsync() and henceforth also run()
     *
     * It seems like this could be a smaller bottleneck for a large amount of
     * kernels. For only 30'000 kernels with each only ten floats or integers,
     * i.e. a total of roughly 1 MB to send, may be negligible compared to
     * the device-to-host copy latency.
     */
    private void writeBlocksList( List<Kernel> work )
    {
        final Stopwatch watch = new Stopwatch();
        watch.start();

        /* no branching on m_usingHandles ? */
        /* setAddress(0) doesn't change the heapEnd, meaning the resulting
         * heapEnd will be the maximum of the last and the one after writing
         * everything in this method */
        m_objectMemory.clearHeapEndPtr();
        m_handlesMemory.setAddress(0);

        final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );
        serializer.writeStaticsToHeap();

        for ( Kernel kernel : work )
            m_handlesMemory.writeRef( serializer.writeToHeap( kernel ) );
        m_objectMemory.align16();

        if ( Configuration.getPrintMem() )
        {
            final BufferPrinter printer = new BufferPrinter();
            printer.print(m_objectMemory, 0, 256);
        }

        watch.stop();
        m_stats.setSerializationTime( watch.elapsedTimeMillis() );
    }

    /**
     * Calls and times cudaLaunch JNI method
     **/
    private void runGpu()
    {
        final Stopwatch watch = new Stopwatch();
        watch.start();
            cudaRun( m_nativeContext, m_objectMemory, !m_usingHandles ? 1 : 0, m_stats );
        watch.stop();
        m_stats.setExecutionTime( watch.elapsedTimeMillis() );

        m_requiredMemorySize = m_objectMemory.getHeapEndPtr();
    }

    private void readBlocksSetup( final Serializer serializer )
    {
        m_readBlocksStopwatch.start();

        m_objectMemory.setAddress(0);
        m_exceptionsMemory.setAddress(0);

        if ( Configuration.runtimeInstance().getExceptions() )
        {
            /* for each thread in the kernel get its corresponding exception
             * and evaluate i.e. throw it, if necessary */
            for( long i = 0; i < m_threadConfig.getNumThreads(); ++i )
            {
                final long ref = m_exceptionsMemory.readRef();
                if ( ref != 0 )
                {
                    final long ref_num = ref >> 4; /* = ref / 16 (?) */
                    if ( ref_num == m_compiledKernel.getNullPointerNumber() ) {
                        throw new NullPointerException("Null pointer exception while running on GPU");
                    } else if ( ref_num == m_compiledKernel.getOutOfMemoryNumber() ) {
                        throw new OutOfMemoryError("Out of memory error while running on GPU");
                    }

                    m_objectMemory.setAddress(ref);
                    final Object except = serializer.readFromHeap(null, true, ref);
                    if ( except instanceof Error )
                    {
                        Error except_th = (Error) except;
                        throw except_th;
                    }
                    else if ( except instanceof GpuException )
                    {
                        GpuException gpu_except = (GpuException) except;
                        throw new ArrayIndexOutOfBoundsException(
                            "[CUDAContext.java]\n" +
                            "array_index  : " + gpu_except.m_arrayIndex  + "\n" +
                            "array_length : " + gpu_except.m_arrayLength + "\n" +
                            "array        : " + gpu_except.m_array       + "\n"
                        );
                    } else {
                        throw new RuntimeException( (Throwable) except );
                    }
                }
            }
        }

        serializer.readStaticsFromHeap();
    }

    private void readBlocksTemplate()
    {
        final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );
        readBlocksSetup(serializer);
        m_handlesMemory.setAddress(0);

        final long handle = m_handlesMemory.readRef();
        serializer.readFromHeap( m_compiledKernel, true, handle );

        if ( Configuration.getPrintMem() )
        {
            final BufferPrinter printer = new BufferPrinter();
            printer.print(m_objectMemory, 0, 256);
        }
        /* the start seems to be in readBlocksSetup ??? */
        m_readBlocksStopwatch.stop();
        m_stats.setDeserializationTime( m_readBlocksStopwatch.elapsedTimeMillis() );
    }

    public void readBlocksList( final List<Kernel> kernelList )
    {
        final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );
        readBlocksSetup(serializer);
        m_handlesMemory.setAddress(0);

        for ( Kernel kernel : kernelList )
        {
            final long ref = m_handlesMemory.readRef();
            serializer.readFromHeap( kernel, true, ref );
        }

        if ( Configuration.getPrintMem() )
        {
            final BufferPrinter printer = new BufferPrinter();
            printer.print( m_objectMemory, 0, 256 );
        }
        /* the start seems to be in readBlocksSetup ??? */
        m_readBlocksStopwatch.stop();
        m_stats.setDeserializationTime( m_readBlocksStopwatch.elapsedTimeMillis() );
    }

    private static native void initializeDriver();
    private native long allocateNativeContext();
    private native void freeNativeContext(long nativeContext);

    private native void nativeBuildState
    (
        long   nativeContext  ,
        int    deviceIndex    ,
        byte[] cubinFile      ,
        int    cubinLength    ,
        int    threadCountX   ,
        int    threadCountY   ,
        int    threadCountZ   ,
        int    blockCountX    ,
        int    blockCountY    ,
        int    numThreads     ,
        Memory objectMem      ,
        Memory handlesMem     ,
        Memory exceptionsMem  ,
        Memory classMem       ,
        int    usingExceptions,
        int    cacheConfig
    );

    private native void cudaRun
    (
        long     nativeContext,
        Memory   objectMem,
        int      usingKernelTemplates,
        StatsRow stats
    );

}
