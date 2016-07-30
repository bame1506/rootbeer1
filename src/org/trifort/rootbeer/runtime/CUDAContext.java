
package org.trifort.rootbeer.runtime;


import java.util.List;
import java.util.Arrays;
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
    /* if true activates debug output for this class */
    private final static boolean         debugging = true      ;

    private final GpuDevice              m_gpuDevice           ;
    private final boolean                m_is32bit             ;

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
    private final StatsRow               m_stats               ;

    private final ExecutorService        m_exec                ;
    private final Disruptor   <GpuEvent> m_disruptor           ;
    private final EventHandler<GpuEvent> m_handler             ;
    private final RingBuffer  <GpuEvent> m_ringBuffer          ;

    /* This is a static constructor for initializing all static members once.
     * Unlike the constructor this function is only called once on program
     * start (or first use of this class) and not on each new object creation.
     * This also means when initializing multiple GPU devices this function
     * is only called once. */
    static { initializeDriver(); }

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
        m_handler              = (EventHandler<GpuEvent>) new GpuEventHandler();
        m_disruptor.handleEventsWith( m_handler );
        m_ringBuffer           = m_disruptor.start();
        /* the started event handler will exist as long as the object to this
         * class */

        m_gpuDevice            = device;
        m_memorySize           = -1;    /* automatically determine size */

        final String arch      = System.getProperty( "os.arch" );
        m_is32bit              = arch.equals( "x86" ) || arch.equals( "i386" );

        m_usingUncheckedMemory = true;
        m_usingHandles         = false;
        m_nativeContext        = allocateNativeContext();
        m_cacheConfig          = CacheConfig.PREFER_NONE;

        m_stats                = new StatsRow();
    }

    @Override public void close()
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

    @Override public void setKernel( final Kernel kernelTemplate )
    {
        m_kernelTemplate = kernelTemplate;
        m_compiledKernel = (CompiledKernel) kernelTemplate;
    }

    /* Just a method which emulates default arguments */
    @Override public void setThreadConfig
    (
        final int threadCountX,
        final int blockCountX ,
        final int numThreads
    )
    {
        setThreadConfig( threadCountX, 1, 1, blockCountX, 1, numThreads );
    }
    /* Just a method which emulates default arguments */
    @Override public void setThreadConfig
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
    @Override public void setThreadConfig
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
    @Override public void buildState()
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
            if ( m_usingHandles )
                m_handlesMemory  = new FixedMemory(4*m_threadConfig.getNumThreads());
            else
                m_handlesMemory  = new FixedMemory(4);
        }
        else
        {
            /* exactly same as FixedMemory block above */
            m_classMemory        = new CheckedFixedMemory(1024);
            m_exceptionsMemory   = new CheckedFixedMemory(getExceptionsMemSize(m_threadConfig));
            m_textureMemory      = new CheckedFixedMemory(8);
            if ( m_usingHandles )
                m_handlesMemory  = new CheckedFixedMemory(4*m_threadConfig.getNumThreads());
            else
                m_handlesMemory  = new CheckedFixedMemory(4);
        }
        /* findMemory size needs m_classMemory, m_textureMemory to be set! */
        if ( m_memorySize == -1 )
            findMemorySize( );
        if ( m_usingUncheckedMemory )
            m_objectMemory = new FixedMemory( m_memorySize );
        else
            m_objectMemory = new CheckedFixedMemory( m_memorySize );

        /* push a new command into the multithreaded com.lmax ring buffer */
        final long seq = m_ringBuffer.next();
        final GpuEvent gpuEvent = m_ringBuffer.get( seq );
        gpuEvent.setValue( GpuEventCommand.NATIVE_BUILD_STATE );
        gpuEvent.getFuture().reset();
        m_ringBuffer.publish( seq );
        /* wait for NATIVE_BUILD_STATE to have finished */
        gpuEvent.getFuture().take();
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
        if ( debugging )
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

  @Override public void run()
  {
      GpuFuture future = runAsync();
      future.take();
  }

  /**
   * @todo Is this and run without a list actually being used?
   * Normally Rootbeer.java:run will be used and that only works with a
   * list of kernels
   */
  @Override public GpuFuture runAsync()
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
    @Override public GpuFuture runAsync( final List<Kernel> work )
    {
        final long seq = m_ringBuffer.next();
        final GpuEvent gpuEvent = m_ringBuffer.get( seq );
            gpuEvent.setKernelList( work );
            gpuEvent.setValue( GpuEventCommand.NATIVE_RUN_LIST );
        gpuEvent.getFuture().reset();
        m_ringBuffer.publish( seq );
        return gpuEvent.getFuture();
    }

    @Override public void run( final List<Kernel> work )
    {
        if ( debugging )
            System.out.println( "[CUDAContext.java:run(List<Kernel>)] calling runAsync" );
        GpuFuture future = runAsync( work );
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
            try
            {
                switch ( gpuEvent.getValue() )
                {
                    case NATIVE_BUILD_STATE:
                    {
                        final boolean usingExceptions = Configuration.runtimeInstance().getExceptions();
                        nativeBuildState(
                            m_nativeContext                 ,
                            m_gpuDevice.getDeviceId()       ,
                            m_cubinFile                     ,
                            m_cubinFile.length              ,
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

    /* @see writeBlocksList( List<Kernel> work ) */
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

        final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );
        serializer.writeStaticsToHeap(); // writes statics to m_objectMemory

        m_handlesMemory.writeRef( serializer.writeToHeap( m_compiledKernel ) );
        m_objectMemory.align16();

        if ( Configuration.getPrintMem() )
            BufferPrinter.print( m_objectMemory, 0, 256 );

        watch.stop();
        m_stats.setSerializationTime( watch.elapsedTimeMillis() );
    }

    private void readBlocksTemplate()
    {
        final Stopwatch watch = new Stopwatch();
        watch.start();

        m_objectMemory.setAddress(0);
        m_handlesMemory.setAddress(0);
        final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );

        /* @todo shouldn't this only be done if exceptions are activated ???
         * which would make it possble to merge it into the submethod */
        m_exceptionsMemory.setAddress(0);
        if ( Configuration.runtimeInstance().getExceptions() )
            readBlocksGetAndCheckExceptions( serializer );

        serializer.readFromHeap( m_compiledKernel, true, m_handlesMemory.readRef() );

        /* Debug output of heap m_objectMemory */
        if ( Configuration.getPrintMem() )
            BufferPrinter.print( m_objectMemory, 0, 256 );

        watch.stop();
        m_stats.setDeserializationTime( watch.elapsedTimeMillis() );
    }

    /**
     * Format a value like 13941672 to '13 MiB 302 kiB 936 B'
     */
    private static String formatSize( long nBytes )
    {
        final int factor = 1024;
        final List<String> units = Arrays.asList( "B", "kiB", "MiB", "GiB", "TiB" );
        String ret = "";
        for ( int i = 0; i < units.size(); ++i )
        {
            ret = ( nBytes % factor ) + " " + units.get(i) + " " + ret;
            if ( ( nBytes /= factor ) == 0 )
                break;
        }
        return ret;
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
    private void writeBlocksList( final List<Kernel> work )
    {
        final Stopwatch watch = new Stopwatch();
        watch.start();

        String output = "";

        try
        {

            /* no branching on m_usingHandles ? */
            /* setAddress(0) doesn't change the heapEnd, meaning the resulting
             * heapEnd will be the maximum of the last and the one after writing
             * everything in this method */
            m_objectMemory.clearHeapEndPtr();
            m_handlesMemory.setAddress(0);

            if ( debugging )
            {
                output +=
                    "\n[CUDAContext.java:writeBlocksList]\n" +
                    "|  m_objectMemory  current address: " + m_objectMemory .getPointer() + "\n" +
                    "|  m_handlesMemory current address: " + m_handlesMemory.getPointer() + "\n";
            }

            final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );
            serializer.writeStaticsToHeap();  // writes statics to m_objectMemory

            if ( debugging )
            {
                output +=
                    "\n[CUDAContext.java:writeBlocksList]" + "\n" +
                    "| After writing statics to heap"    + "\n" +
                    "|   m_objectMemory  current address: " + m_objectMemory .getPointer() + "\n" +
                    "|   m_handlesMemory current address: " + m_handlesMemory.getPointer() + "\n";
            }

            /* this writes each kernel and their non-static members to object
             * memory and saves the returned manual/relative pointer for the
             * heap byte array to m_handlesMemory so that each kernel can be
             * refound on GPU.
             *    m_handlesMemory will increase by work.size() * 4 Bytes
             *    m_objectMemory increases by work.size() * (sum of all members
             *      kernel) + one-time members e.g. the array where the private
             *      members point to
             *
             * Example Output for CountKernel:
             *   [CUDAContext.java:writeBlocksList]
             *   | Writing the first kernel needed : 196720
             *   | Every consequent kernel needed  : 48
             *   | => one-time kernel code needs   : 196672
             * The one-time code is exactly work.size()*16 + 64. This seems to
             * The kernel contains two long arrays of length work.size() (=12288)
             * This means each array element seems to need twice the size Oo?
             */
            final long nPreFirstKernel = m_objectMemory.getPointer();
            long nPostFirstKernel = -1;
            for ( final Kernel kernel : work )
            {
                /* the bug to rule them all at some runs also appears here in
                 * form of a NullPointerException in the writeToHeap method,
                 * which writes to m_objectMemory. Note that CheckedFixedMemory
                 * was used, so such things should normally be found by that ...
                 *
                 * java.lang.NullPointerException
                 *   at org.trifort.rootbeer.runtime.Serializer.checkWriteCache(Serializer.java:97)
                 *   at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:134)
                 *   at MonteCarloPiKernel.org_trifort_writeToHeapRefFields_MonteCarloPiKernel0(Jasmin)
                 *   at MonteCarloPiKernelSerializer.doWriteToHeap(Jasmin)
                 *   at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:144)
                 *   at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:47)
                 *   at org.trifort.rootbeer.runtime.CUDAContext.writeBlocksList(CUDAContext.java:599)
                 *   at org.trifort.rootbeer.runtime.CUDAContext.access$1400(CUDAContext.java:28)
                 *   at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:437)
                 *   at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:387)
                 *   at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
                 *   at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
                 *   at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
                 *   at java.lang.Thread.run(Thread.java:724)
                 */
                m_handlesMemory.writeRef( serializer.writeToHeap( kernel ) );
                if ( nPostFirstKernel == -1 )
                    nPostFirstKernel = m_objectMemory.getPointer();
            }
            final long nPostLastKernel = m_objectMemory.getPointer();

            if ( debugging )
            {
                final long nBytesPerKernel = ( nPostLastKernel - nPostFirstKernel ) / ( work.size() - 1 );
                output +=
                    "\n[CUDAContext.java:writeBlocksList]\n" +
                    "| Writing the first kernel needed : " + ( nPostFirstKernel - nPreFirstKernel ) + " B\n" +
                    "| Every consequent kernel needed  : " + nBytesPerKernel + " B\n" +
                    "| => one-time kernel code needs   : " +
                    ( nPostFirstKernel - nPreFirstKernel - nBytesPerKernel ) + " B\n";
                assert( ( nPostLastKernel - nPostFirstKernel ) % ( work.size() - 1 ) == 0 );
            }

            m_objectMemory.align16();

            if ( debugging )
            {
                output +=
                    "\n[CUDAContext.java:writeBlocksList]\n" +
                    "| After align16 call:\n" +
                    "|   m_objectMemory  current address: " + m_objectMemory .getPointer() + "\n" +
                    "|   m_handlesMemory current address: " + m_handlesMemory.getPointer() + "\n";
            }

            if ( debugging && ! Configuration.getPrintMem() )
            {
                /* @todo For some reason the size returned by getSize is different
                 * on each different compilation Oo ? But it stays the same on
                 * different runs with the same binary. Is something like the time
                 * encoded Oo? ??? */
                output +=
                    "\n[CUDAContext.java:writeBlocksList] After writing here " +
                    "are the first 1024 Bytes of m_objectMemory (" +
                    m_objectMemory.getSize() + " B = " +
                    formatSize( m_objectMemory.getSize() ) + ") :\n" +
                    BufferPrinter.toString( m_objectMemory , 0, 1024 ) +
                    "\n[CUDAContext.java:writeBlocksList] and also the " +
                    "first 1024 Bytes of m_handlesMemory (" + m_handlesMemory.getSize() + " B = " +
                    formatSize( m_handlesMemory.getSize() ) + ") :\n" +
                    BufferPrinter.toString( m_handlesMemory, 0, 1024 );
            }

            if ( Configuration.getPrintMem() )
                BufferPrinter.print( m_objectMemory, 0, 256 );

        }
        finally
        {
            System.out.println( output );
        }

        watch.stop();
        m_stats.setSerializationTime( watch.elapsedTimeMillis() );
    }

    /**
     * This function should do the exact inverse of writeBlocksList !
     */
    public void readBlocksList( final List<Kernel> work )
    {
        final Stopwatch watch = new Stopwatch();
        watch.start();

        long iKernel = 0;
        long ref     = 0;
        String output = "";

        try {
            m_objectMemory .setAddress(0);
            m_handlesMemory.setAddress(0);
            final Serializer serializer = m_compiledKernel.getSerializer( m_objectMemory, m_textureMemory );

            /* @todo shouldn't this only be done if exceptions are activated ???
             * which would make it possble to merge it into the submethod */
            m_exceptionsMemory.setAddress(0);
            if ( Configuration.runtimeInstance().getExceptions() )
                readBlocksGetAndCheckExceptions( serializer );

            if ( debugging )
            {
                output +=
                    "\n[CUDAContext.java:readBlocksList]\n" +
                    "|  m_objectMemory  current address: " + m_objectMemory .getPointer() + "\n" +
                    "|  m_handlesMemory current address: " + m_handlesMemory.getPointer() + "\n" +
                    "|  m_objectMemory  size           : " + m_objectMemory .getSize() +
                    " B = " + formatSize( m_objectMemory .getSize() ) + "\n" +
                    "|  m_handlesMemory size           : " + m_handlesMemory.getSize() +
                    " B = " + formatSize( m_handlesMemory.getSize() ) + "\n";
            }

            serializer.readStaticsFromHeap();

            if ( debugging )
            {
                output +=
                    "\n[CUDAContext.java:readBlocksList]\n" +
                    "|After reading statics from heap\n" +
                    "|  m_objectMemory  current address: " + m_objectMemory .getPointer() + "\n" +
                    "|  m_handlesMemory current address: " + m_handlesMemory.getPointer() + "\n";
            }

            final long nPreFirstKernel = m_objectMemory.getPointer();
            long nPostFirstKernel  = -1;
            long nPostSecondKernel = -1;
            for ( final Kernel kernel : work )
            {
                ref = m_handlesMemory.readRef();
                // inverse: m_handlesMemory.writeRef( serializer.writeToHeap( kernel ) );
                serializer.readFromHeap( kernel, true, ref );

                if ( debugging )
                {
                    if ( nPostSecondKernel == -1 && nPostFirstKernel != -1 ) // so only in second run
                    {
                        output += "[CUDAContext.java:readBlocksList] read from ref " +
                            ref + ", now pointer is at " + m_objectMemory.getPointer() + "\n";
                        nPostSecondKernel = ref;
                    }
                    else if ( nPostFirstKernel == -1 )
                    {
                        output += "[CUDAContext.java:readBlocksList] read from ref " +
                            ref + ", now pointer is at " + m_objectMemory.getPointer() + "\n";
                        nPostFirstKernel = ref; //m_objectMemory.getPointer();
                    }
                    else if ( iKernel % 40 == 0 )
                    {
                        output += "\nLast read Kernel " + iKernel +
                            " from heap ref " + ref + " (handle pointer " +
                            m_handlesMemory.getPointer() + ") ";
                    }
                    else
                        output += ".";
                    iKernel++;
                }
            }
            final long nPostLastKernel = m_objectMemory.getPointer();

            if ( debugging )
            {
                final long nBytesPerKernel = ( nPostLastKernel - nPostFirstKernel ) / ( work.size() - 1 );
                output +=
                    "\n[CUDAContext.java:readBlocksList]\n" +
                    "| Reading the first  kernel from address : " + nPostFirstKernel  + "\n" +
                    "| Reading the second kernel from address : " + nPostSecondKernel + "\n" +
                    "| Every consequent kernel needed  : " + nBytesPerKernel + " B"   + "\n" +
                    "| => one-time kernel code needs   : " +
                    ( nPostFirstKernel - nPreFirstKernel - nBytesPerKernel) + " B";
                /* memory dump */
                if ( ! Configuration.getPrintMem() )
                {
                    output +=
                        "\n[CUDAContext.java:readBlocksList] After reading here " +
                        "are the first 1024 Bytes of m_objectMemory (" + m_objectMemory.getSize() + " B = " +
                        formatSize( m_objectMemory.getSize() ) + ") :\n" +
                        BufferPrinter.toString( m_objectMemory , 0, 1024 ) + "\n" +
                        "\n[CUDAContext.java:readBlocksList] After reading here " +
                        "are the first 1024 Bytes of m_handlesMemory (" + m_handlesMemory.getSize() + " B = " +
                        formatSize( m_handlesMemory.getSize() ) + ") :\n" +
                        BufferPrinter.toString( m_handlesMemory, 0, 1024 ) + "\n";
                }
            }
        }
        catch ( Exception e )
        {
            output += "\n!!! [CUDAContext.java:readBlocksList] " +
                "Exception occured when trying to read kernel " + iKernel +
                " out of " + work.size() + " kernels " +
                "from heap (relative address / saved handle: " + ref + ")\n" +
                "m_objectMemory at that address and before and after:\n" +
                BufferPrinter.toString( m_objectMemory, Math.max( 0, ref-256 ), 2*256) + "\n";
            /**
             *  private long[] mnHits;
             *  private long[] mnIterations;
             *  private int    miLinearThreadId;
             *  private long   mRandomSeed;
             *  private long   mnDiceRolls;
             *
             * Sample Output:
             *
             * !!! [CUDAContext.java:readBlocksList] Exception occured when
             * trying to read kernel 167 from heap (relative address /
             * saved handle: 862896)
             *
             *  862768 : 02 00 00 00  ca 0c 00 00  40 00 00 00  00 00 00 00
             *  862784 : ff ff ff ff  00 00 00 00  00 00 00 00  00 00 00 00
             *  862800 : 4b 6a 00 00  4d 9e 00 00  fc b7 f6 9d  d8 89 65 00
             *  862816 : b2 13 00 00  00 00 00 00  a5 00 00 00  00 00 00 00
             *           48 Bytes of data which is the same in all kernels
             *          +---------------------------------------------------+
             *  862832 :|02           ca 0c        40                       |
             *  862848 :|ff ff ff ff              __________________________+
             *  862864 :|4b 6a        4d 9e      | d4 41 94 76  62 27 66
             *          +------------------------+ very probably mRandomSeed
             *          ( note how the highest byte i.e. 66 or 65 or 68 grows
             *            only very slowly (and surely lineraly) )
             *  862880 : b2 13                     a6
             *           +----------------------+  +----------------------+
             *    13b1 = ceil( 134217728 / 26624 )  = 166 (miLinearThreadId)
             *        (mnDiceRolls) (64 Bit)
             *
             *  862896 : 02           ca 0c        40
             *  862912 : ff ff ff ff
             *  862928 : 4b 6a        4d 9e        ac cb 31 4f  ec c4 66
             *  862944 : b2 13                     a7
             *
             *  863024 : 02 00 00 00  ca 0c 00 00  40 00 00 00  00 00 00 00
             *  863040 : ff ff ff ff  00 00 00 00  00 00 00 00  00 00 00 00
             *  863056 : 4b 6a 00 00  4d 9e 00 00  5c df 6c 00  00 00 68 00
             *  863072 : b2 13 00 00  00 00 00 00  a9 00 00 00  00 00 00 00
             *
             * [MonteCarloPi.scala:calc] iterations actually done : 134217728

             *  @todo The data after 862896 looks perfectly normal to me !!!
             *  I don't understand why suddenly the exception occurs:
             *
             *  MonteCarloPiKernel cannot be cast to [J
             *    at MonteCarloPiKernel.org_trifort_readFromHeapRefFields_MonteCarloPiKernel0(Jasmin)
             *    at MonteCarloPiKernelSerializer.doReadFromHeap(Jasmin)
             *    at org.trifort.rootbeer.runtime.Serializer.readFromHeap(Serializer.java:188)
             *    at org.trifort.rootbeer.runtime.CUDAContext.readBlocksList(CUDAContext.java:713)
             *    ...
             */
            System.out.print( output );
            e.printStackTrace(); // don't throw exception, because Spark would restart then infinitely often
            throw e;
        }

        /* debugging output */
        if ( Configuration.getPrintMem() )
            BufferPrinter.print( m_objectMemory, 0, 256 );

        watch.stop();
        m_stats.setDeserializationTime( watch.elapsedTimeMillis() );
    }

    private void readBlocksGetAndCheckExceptions( final Serializer serializer )
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
                    throw new NullPointerException( "Null pointer exception while running on GPU" );
                } else if ( ref_num == m_compiledKernel.getOutOfMemoryNumber() ) {
                    throw new OutOfMemoryError( "Out of memory error while running on GPU" );
                }

                /* won't this setting of m_objectMemory confuse the reading
                 * of further elements ? I guess it doesn't matter, because
                 * in any case an exception is thrown and the logic in here is
                 * exited */
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
                } else
                    throw new RuntimeException( (Throwable) except );
            }
        }
    }

    /**
     * Calls and times cudaLaunch JNI method
     **/
    private void runGpu()
    {
        if ( debugging )
            System.out.println( "[CUDAContext.java:runGpu] execute cudaRun()" );

        final Stopwatch watch = new Stopwatch();
        watch.start();

            cudaRun( m_nativeContext, m_objectMemory, !m_usingHandles ? 1 : 0, m_stats );

        watch.stop();
        m_stats.setExecutionTime( watch.elapsedTimeMillis() );

        m_requiredMemorySize = m_objectMemory.getHeapEndPtr();
    }

    /************** Declarations which will be implemented using **************
     **************     the java native interface from C++       **************/
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
