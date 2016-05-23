package org.trifort.rootbeer.runtime;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.runtime.util.Stopwatch;
import org.trifort.rootbeer.runtimegpu.GpuException;
import org.trifort.rootbeer.util.ResourceReader;

import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.dsl.Disruptor;

import org.trifort.rootbeer.generate.bytecode.Constants;

public class CUDAContext implements Context {

  final private GpuDevice gpuDevice;
  final private boolean is32bit;

  private long           nativeContext       ;
  private long           memorySize          ;  /**< in bytes */
  private byte[]         cubinFile           ;
  private Memory         objectMemory        ;
  private Memory         handlesMemory       ;
  private Memory         textureMemory       ;
  private Memory         exceptionsMemory    ;
  private Memory         classMemory         ;
  private boolean        usingUncheckedMemory;
  private long           requiredMemorySize  ;
  private CacheConfig    cacheConfig         ;
  private ThreadConfig   threadConfig        ;
  private Kernel         kernelTemplate      ;
  private CompiledKernel compiledKernel      ;
  private boolean        usingHandles        ;

  final private StatsRow stats;
  final private Stopwatch writeBlocksStopwatch;
  final private Stopwatch runStopwatch;
  final private Stopwatch runOnGpuStopwatch;
  final private Stopwatch readBlocksStopwatch;

  final private ExecutorService exec;
  final private Disruptor<GpuEvent> disruptor;
  final private EventHandler<GpuEvent> handler;
  final private RingBuffer<GpuEvent> ringBuffer;

  /* ??? who calls this */
  static {
      initializeDriver();
  }

  public CUDAContext( GpuDevice device )
  {
      exec = Executors.newCachedThreadPool(new ThreadFactory() {
        public Thread newThread(Runnable r) {
          Thread t = new Thread(r);
          t.setDaemon(true);
          return t;
        }
      });
      disruptor  = new Disruptor<GpuEvent>(GpuEvent.EVENT_FACTORY, 64, exec);
      handler    = new GpuEventHandler();
      disruptor.handleEventsWith(handler);
      ringBuffer = disruptor.start();
      gpuDevice  = device;
      memorySize = -1;    /* automatically determine size */

      String arch = System.getProperty("os.arch");
      is32bit     = arch.equals("x86") || arch.equals("i386");

      usingUncheckedMemory = true;
      usingHandles         = false;
      nativeContext        = allocateNativeContext();
      cacheConfig          = CacheConfig.PREFER_NONE;

      stats                = new StatsRow();
      writeBlocksStopwatch = new Stopwatch();
      runStopwatch         = new Stopwatch();
      runOnGpuStopwatch    = new Stopwatch();
      readBlocksStopwatch  = new Stopwatch();
  }

  @Override
  public GpuDevice getDevice() {
    return gpuDevice;
  }

  @Override
  public void close() {
    disruptor.shutdown();
    exec.shutdown();
    freeNativeContext(nativeContext);

    if(objectMemory != null){
      objectMemory.close();
    }
    if(handlesMemory != null){
      handlesMemory.close();
    }
    if(exceptionsMemory != null){
      exceptionsMemory.close();
    }
    if(classMemory != null){
      classMemory.close();
    }
    if(textureMemory != null){
      textureMemory.close();
    }
  }

  @Override
  public void setMemorySize( long memorySize ) {
      this.memorySize = memorySize;
  }

  @Override
  public void setKernel(Kernel kernelTemplate) {
    this.kernelTemplate = kernelTemplate;
    this.compiledKernel = (CompiledKernel) kernelTemplate;
  }

  @Override
  public void setCacheConfig(CacheConfig cacheConfig) {
    this.cacheConfig = cacheConfig;
  }

  @Override
  public void setUsingHandles(boolean value){
    usingHandles = value;
  }

  @Override
  public void useCheckedMemory(){
    this.usingUncheckedMemory = false;
  }

  @Override
  public void setThreadConfig(ThreadConfig threadConfig) {
    this.threadConfig = threadConfig;
  }

  @Override
  public void setThreadConfig(int threadCountX, int blockCountX,
      int numThreads) {
    setThreadConfig(threadCountX, 1, 1, blockCountX, 1, numThreads);
  }

  @Override
  public void setThreadConfig(int threadCountX, int threadCountY,
      int blockCountX, int blockCountY, int numThreads) {
    setThreadConfig(threadCountX, threadCountY, 1, blockCountX, blockCountY, numThreads);
  }

  @Override
  public void setThreadConfig(int threadCountX, int threadCountY,
      int threadCountZ, int blockCountX, int blockCountY,
      int numThreads) {
    this.threadConfig = new ThreadConfig(threadCountX, threadCountY, threadCountZ,
        blockCountX, blockCountY, numThreads);
  }

  /* Seems to load cubin file and allocates memory for member 'compiledKernel',
   * therefore 'setKernel' must be called prior to this! */
  @Override
  public void buildState()
  {
      String filename;
      int size = 0;
      boolean error = false;

      if(is32bit){
          filename = compiledKernel.getCubin32();
          size     = compiledKernel.getCubin32Size();
          error    = compiledKernel.getCubin32Error();
      } else {
          filename = compiledKernel.getCubin64();
          size     = compiledKernel.getCubin64Size();
          error    = compiledKernel.getCubin64Error();
      }

      if(error){
          throw new RuntimeException("CUDA code compiled with error");
      }

      cubinFile = readCubinFile(filename, size);

      if ( usingUncheckedMemory )
      {
          classMemory        = new FixedMemory(1024);
          exceptionsMemory   = new FixedMemory(getExceptionsMemSize(threadConfig));
          textureMemory      = new FixedMemory(8);
          if(usingHandles){
              handlesMemory  = new FixedMemory(4*threadConfig.getNumThreads());
          } else {
              handlesMemory  = new FixedMemory(4);
          }
      }
      else
      {
          exceptionsMemory   = new CheckedFixedMemory(getExceptionsMemSize(threadConfig));
          classMemory        = new CheckedFixedMemory(1024);
          textureMemory      = new CheckedFixedMemory(8);
          if(usingHandles){
              handlesMemory  = new CheckedFixedMemory(4*threadConfig.getNumThreads());
          } else {
              handlesMemory  = new CheckedFixedMemory(4);
          }
      }
      if ( memorySize == -1 ) {
          findMemorySize(cubinFile.length);
      }
      if(usingUncheckedMemory){
          objectMemory = new FixedMemory(memorySize);
      }   else {
          objectMemory = new CheckedFixedMemory(memorySize);
      }

      long seq = ringBuffer.next();
      GpuEvent gpuEvent = ringBuffer.get(seq);
      gpuEvent.setValue(GpuEventCommand.NATIVE_BUILD_STATE);
      gpuEvent.getFuture().reset();
      ringBuffer.publish(seq);
      gpuEvent.getFuture().take();
  }

    private long getExceptionsMemSize( ThreadConfig thread_config )
    {
        if ( Configuration.runtimeInstance().getExceptions() ) {
            return 4L*thread_config.getNumThreads();
        } else {
            return 4;
        }
    }

    /**
     * @todo why not call getRessourceArray directly? This function does not
     *       add any value.
     */
    private byte[] readCubinFile( String filename, int length )
    {
        try
        {
            byte[] buffer = ResourceReader.getResourceArray(filename, length);
            return buffer;
        }
        catch(Exception ex)
        {
	       ex.printStackTrace();
           throw new RuntimeException(ex);
        }
    }

    /**
     * Automatically finds a good memory size needed for allocation from
     * several parameters. Also checks if the free memory is enough to hold it.
     *
     * @param[in] cubinFileLength file size in bytes
     */
    private void findMemorySize( int cubinFileLength )
    {
        final long freeMemSizeGPU = gpuDevice.getFreeGlobalMemoryBytes();
        final long freeMemSizeCPU = Runtime.getRuntime().freeMemory();

        /* in the worst case classMemory is only bytes which all would get
         * aligned to 16-byte boundarys (Constants.MallocAlignBytes) resulting
         * in 16-fold memory needed. Exception size are assumed to be 4 bytes
         * and also assumed to be aligned (are they???) */
        final long neededMemory =
            cubinFileLength + Constants.MallocAlignBytes +
            exceptionsMemory.getSize() / 4 * Constants.MallocAlignBytes +
            classMemory.getSize() * Constants.MallocAlignBytes;

        final String debugOutput =
            "  Debugging Output:\n"                                              +
            "    GPU size         : " + freeMemSizeGPU                  + " B\n" +
            "    CPU_SIZE         : " + freeMemSizeCPU                  + " B\n" +
            "    Exceptions size  : " + exceptionsMemory.getSize()      + " B\n" +
            "    class memory size: " + classMemory.getSize()           + " B\n" +
            "    cubin size       : " + cubinFileLength                 + " B\n" +
            "    cubin32 size     : " + compiledKernel.getCubin32Size() + " B\n" +
            "    cubin64 size     : " + compiledKernel.getCubin64Size() + " B\n" +
        //    "    kernel.doGetSize : " + serial.doGetSize(compiledKernel) + " B\n" +
            "    alignment        : " + Constants.MallocAlignBytes      + " B\n" ;
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
        memorySize = neededMemory;
    }

  @Override
  public long getRequiredMemory() {
    return requiredMemorySize;
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
      long seq = ringBuffer.next();
      GpuEvent gpuEvent = ringBuffer.get(seq);
          gpuEvent.setValue(GpuEventCommand.NATIVE_RUN);
      gpuEvent.getFuture().reset();
      ringBuffer.publish(seq);
      return gpuEvent.getFuture();
  }

  /**
   * Launches a kernel asynchronously
   */
  @Override
  public GpuFuture runAsync(List<Kernel> work)
  {
      long seq = ringBuffer.next();
      GpuEvent gpuEvent = ringBuffer.get(seq);
          gpuEvent.setKernelList(work);
          gpuEvent.setValue(GpuEventCommand.NATIVE_RUN_LIST);
      gpuEvent.getFuture().reset();
      ringBuffer.publish(seq);
      return gpuEvent.getFuture();
  }

  @Override
  public void run(List<Kernel> work) {
    GpuFuture future = runAsync(work);
    future.take();
  }

  @Override
  public StatsRow getStats() {
    return stats;
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
                        boolean usingExceptions = Configuration.runtimeInstance().getExceptions();
                        nativeBuildState( nativeContext, gpuDevice.getDeviceId(), cubinFile,
                            cubinFile.length,
                            threadConfig.getThreadCountX(),
                            threadConfig.getThreadCountY(),
                            threadConfig.getThreadCountZ(),
                            threadConfig.getBlockCountX (),
                            threadConfig.getBlockCountY (),
                            threadConfig.getNumThreads  (),
                            objectMemory, handlesMemory, exceptionsMemory, classMemory,
                            usingExceptions ? 1 : 0, cacheConfig.ordinal() );
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
                        break;
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
         */
        writeBlocksStopwatch.start();
        objectMemory.clearHeapEndPtr();

        /* @todo why isn't this needed in writeBlocksList(List<Kernel> work) */
        handlesMemory.setAddress(0);

        Serializer serializer = compiledKernel.getSerializer(objectMemory, textureMemory);
        serializer.writeStaticsToHeap();

        long handle = serializer.writeToHeap( compiledKernel );
        handlesMemory.writeRef(handle);
        objectMemory.align16();

        if(Configuration.getPrintMem()){
            BufferPrinter printer = new BufferPrinter();
            printer.print(objectMemory, 0, 256);
        }

        writeBlocksStopwatch.stop();
        stats.setSerializationTime(writeBlocksStopwatch.elapsedTimeMillis());
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
        writeBlocksStopwatch.start();
        objectMemory.clearHeapEndPtr();

        Serializer serializer = compiledKernel.getSerializer(objectMemory, textureMemory);
        serializer.writeStaticsToHeap();

        for(Kernel kernel : work){
            long handle = serializer.writeToHeap(kernel);
            handlesMemory.writeRef(handle);
        }
        objectMemory.align16();

        if(Configuration.getPrintMem()){
            BufferPrinter printer = new BufferPrinter();
            printer.print(objectMemory, 0, 256);
        }

        writeBlocksStopwatch.stop();
        stats.setSerializationTime(writeBlocksStopwatch.elapsedTimeMillis());
    }

    /**
     * Calls and times cudaLaunch JNI method
     **/
    private void runGpu()
    {
        runOnGpuStopwatch.start();
        cudaRun(nativeContext, objectMemory, b2i(!usingHandles), stats);
        runOnGpuStopwatch.stop();
        requiredMemorySize = objectMemory.getHeapEndPtr();
        stats.setExecutionTime(runOnGpuStopwatch.elapsedTimeMillis());
    }

  private void readBlocksSetup(Serializer serializer){
    readBlocksStopwatch.start();
    objectMemory.setAddress(0);
    exceptionsMemory.setAddress(0);

    if(Configuration.runtimeInstance().getExceptions()){
      for(long i = 0; i < threadConfig.getNumThreads(); ++i){
        long ref = exceptionsMemory.readRef();
        if(ref != 0){
          long ref_num = ref >> 4;
          if(ref_num == compiledKernel.getNullPointerNumber()){
            throw new NullPointerException("NPE while running on GPU");
          } else if(ref_num == compiledKernel.getOutOfMemoryNumber()){
            throw new OutOfMemoryError("OOM error while running on GPU");
          }

          objectMemory.setAddress(ref);
          Object except = serializer.readFromHeap(null, true, ref);
          if(except instanceof Error){
            Error except_th = (Error) except;
            throw except_th;
          } else if(except instanceof GpuException){
            GpuException gpu_except = (GpuException) except;
            throw new ArrayIndexOutOfBoundsException("array_index: "+gpu_except.m_arrayIndex+
                " array_length: "+gpu_except.m_arrayLength+" array: "+gpu_except.m_array);
          } else {
            throw new RuntimeException((Throwable) except);
          }
        }
      }
    }

    serializer.readStaticsFromHeap();
  }

  private void readBlocksTemplate(){
    Serializer serializer = compiledKernel.getSerializer(objectMemory, textureMemory);
    readBlocksSetup(serializer);
    handlesMemory.setAddress(0);

    long handle = handlesMemory.readRef();
    serializer.readFromHeap(compiledKernel, true, handle);

    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(objectMemory, 0, 256);
    }
    readBlocksStopwatch.stop();
    stats.setDeserializationTime(readBlocksStopwatch.elapsedTimeMillis());
  }

  public void readBlocksList(List<Kernel> kernelList) {
    Serializer serializer = compiledKernel.getSerializer(objectMemory, textureMemory);
    readBlocksSetup(serializer);

    handlesMemory.setAddress(0);
    for(Kernel kernel : kernelList){
      long ref = handlesMemory.readRef();
      serializer.readFromHeap(kernel, true, ref);
    }

    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(objectMemory, 0, 256);
    }
    readBlocksStopwatch.stop();
    stats.setDeserializationTime(readBlocksStopwatch.elapsedTimeMillis());
  }

  private int b2i(boolean value){
    if(value){
      return 1;
    } else {
      return 0;
    }
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
