#About this Fork

This fork adds several small bugfixes, quite some code comments, code reduction and also changes the code style of sighted files to something more similar to e.g. [imresh](https://github.com/ComputationalRadiationPhysics/imresh) i.e. braces on new line, 4 spaces instead of 2 indentation and alignment of similar and especially of boiler-plate code.

#Rootbeer

The Rootbeer GPU Compiler lets you use GPUs from within Java. It allows you to use almost anything from Java on the GPU:

  1. Composite objects with methods and fields
  2. Static and instance methods and fields
  3. Arrays of primitive and reference types of any dimension.

ROOTBEER IS PRE-PRODUCTION BETA. IF ROOTBEER WORKS FOR YOU, PLEASE LET ME KNOW AT PCPRATTS@TRIFORT.ORG

Be aware that you should not expect to get a speedup using a GPU by doing something simple
like multiplying each element in an array by a scalar. Serialization time is a large bottleneck
and usually you need an algorithm that is O(n^2) to O(n^3) per O(n) elements of data.

GPU PROGRAMMING IS NOT EASY, EVEN WITH ROOTBEER. EXPECT TO SPEND A MONTH OPTIMIZING TRIVIAL EXAMPLES.

FEEL FREE TO EMAIL ME FOR DISCUSSIONS BEFORE ATTEMPTING TO USE ROOTBEER

An experienced GPU developer will look at existing code and find places where control can
be transfered to the GPU. Optimal performance in an application will have places with serial
code and places with parallel code on the GPU. At each place that a cut can be made to transfer
control to the GPU, the job needs to be sized for the GPU.

For the best performance, you should be using shared memory (NVIDIA term). The shared memory is
basically a software managed cache. You want to have more threads per block, but this often
requires using more shared memory. If you see the [CUDA Occupancy Calculator](http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls) you can see
that for best occupancy you will want more threads and less shared memory. There is a tradeoff
between thread count, shared memory size and register count. All of these are configurable
using Rootbeer.

## Programming  
<b>Kernel Interface:</b> Your code that will run on the GPU will implement the Kernel interface.
You send data to the gpu by adding a field to the object implementing kernel. `gpuMethod` will access the data.

    package org.trifort.rootbeer.runtime;

    public interface Kernel {
      void gpuMethod();
    }

###Simple Example:
This simple example uses kernel lists and no thread config or context. Rootbeer will create a thread config and select the best device automatically. If you wish to use multiple GPUs you need to pass in a Context.

<b>ScalarAddApp.java:</b>  
See the [example](https://github.com/pcpratts/rootbeer1/tree/master/examples/ScalarAddApp)

```java
package org.trifort.rootbeer.examples.scalaradd;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class ScalarAddApp {

  public void multArray(int[] array){
    List<Kernel> tasks = new ArrayList<Kernel>();
    for(int index = 0; index < array.length; ++index){
      tasks.add(new ScalarAddKernel(array, index));
    }

    Rootbeer rootbeer = new Rootbeer();
    rootbeer.run(tasks);
  }

  private void printArray(String message, int[] array){
    for(int i = 0; i < array.length; ++i){
      System.out.println(message+" array["+i+"]: "+array[i]);
    }
  }

  public static void main(String[] args){
    ScalarAddApp app = new ScalarAddApp();
    int length = 10;
    int[] array = new int[length];
    for(int index = 0; index < array.length; ++index){
      array[index] = index;
    }

    app.printArray("start", array);
    app.multArray(array);
    app.printArray("end", array);
  }
}
```

<b>ScalarAddKernel:</b>

```java
package org.trifort.rootbeer.examples.scalaradd;

import org.trifort.rootbeer.runtime.Kernel;

public class ScalarAddKernel implements Kernel {

  private int[] array;
  private int index;

  public ScalarAddKernel(int[] array, int index){
    this.array = array;
    this.index = index;
  }

  public void gpuMethod(){
    array[index] += 1;
  }
}
```

### High Performance Example - Batcher's Even Odd Sort
See the [example](https://github.com/pcpratts/rootbeer1/tree/master/examples/sort)  
See the [slides](http://trifort.org/ads/index.php/lecture/index/27/)  

<b>GPUSort.java</b>  

```java
package org.trifort.rootbeer.sort;

import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.CacheConfig;
import java.util.List;
import java.util.Arrays;
import java.util.Random;

public class GPUSort {

  private int[] newArray(int size){
    int[] ret = new int[size];

    for(int i = 0; i < size; ++i){
      ret[i] = i;
    }
    return ret;
  }

  public void checkSorted(int[] array, int outerIndex){
    for(int index = 0; index < array.length; ++index){
      if(array[index] != index){
        for(int index2 = 0; index2 < array.length; ++index2){
          System.out.println("array["+index2+"]: "+array[index2]);
        }
        throw new RuntimeException("not sorted: "+outerIndex);
      }
    }
  }

  public void fisherYates(int[] array)
  {
    Random random = new Random();
    for (int i = array.length - 1; i > 0; i--){
      int index = random.nextInt(i + 1);
      int a = array[index];
      array[index] = array[i];
      array[i] = a;
    }
  }

  public void sort(){
    //should have at least 192 threads per SM
    int size = 2048;
    int sizeBy2 = size / 2;
    //int numMultiProcessors = 14;
    //int blocksPerMultiProcessor = 512;
    int numMultiProcessors = 2;
    int blocksPerMultiProcessor = 256;
    int outerCount = numMultiProcessors*blocksPerMultiProcessor;
    int[][] array = new int[outerCount][];
    for(int i = 0; i < outerCount; ++i){
      array[i] = newArray(size);
    }

    Rootbeer rootbeer = new Rootbeer();
    List<GpuDevice> devices = rootbeer.getDevices();
    GpuDevice device0 = devices.get(0);
    //create a context with 4212880 bytes objectMemory.
    //you can leave the 4212880 missing at first to
    //use all available GPU memory. after you run you
    //can call context0.getRequiredMemory() to see
    //what value to enter here
    Context context0 = device0.createContext(4212880);
    //use more die area for shared memory instead of
    //cache. the shared memory is a software defined
    //cache that, if programmed properly, can perform
    //better than the hardware cache
    //see (CUDA Occupancy calculator)[http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls]
    context0.setCacheConfig(CacheConfig.PREFER_SHARED);
    //wire thread config for throughput mode. after
    //calling buildState, the book-keeping information
    //will be cached in the JNI driver
    context0.setThreadConfig(sizeBy2, outerCount, outerCount * sizeBy2);
    //configure to use kernel templates. rather than
    //using kernel lists where each thread has a Kernel
    //object, there is only one kernel object (less memory copies)
    //when using kernel templates you need to differetiate
    //your data using thread/block indexes
    context0.setKernel(new GPUSortKernel(array));
    //cache the state and get ready for throughput mode
    context0.buildState();

    while(true){
      //randomize the array to be sorted
      for(int i = 0; i < outerCount; ++i){
        fisherYates(array[i]);
      }
      long gpuStart = System.currentTimeMillis();
      //run the cached throughput mode state.
      //the data now reachable from the only
      //GPUSortKernel is serialized to the GPU
      context0.run();
      long gpuStop = System.currentTimeMillis();
      long gpuTime = gpuStop - gpuStart;

      StatsRow row0 = context0.getStats();
      System.out.println("serialization_time: "+row0.getSerializationTime());
      System.out.println("execution_time: "+row0.getExecutionTime());
      System.out.println("deserialization_time: "+row0.getDeserializationTime());
      System.out.println("gpu_required_memory: "+context0.getRequiredMemory());
      System.out.println("gpu_time: "+gpuTime);

      for(int i = 0; i < outerCount; ++i){
        checkSorted(array[i], i);
        fisherYates(array[i]);
      }

      long cpuStart = System.currentTimeMillis();
      for(int i = 0; i < outerCount; ++i){
        Arrays.sort(array[i]);
      }
      long cpuStop = System.currentTimeMillis();
      long cpuTime = cpuStop - cpuStart;
      System.out.println("cpu_time: "+cpuTime);
      double ratio = (double) cpuTime / (double) gpuTime;
      System.out.println("ratio: "+ratio);
    }
    //context0.close();
  }

  public static void main(String[] args){
    GPUSort sorter = new GPUSort();
    while(true){
      sorter.sort();
    }
  }
}
```

<b>GPUSortKernel.java</b>

```java
package org.trifort.rootbeer.sort;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;


public class GPUSortKernel implements Kernel {

  private int[][] arrays;

  public GPUSortKernel(int[][] arrays){
    this.arrays = arrays;
  }

  @Override
  public void gpuMethod(){
    int[] array = arrays[RootbeerGpu.getBlockIdxx()];
    int index1a = RootbeerGpu.getThreadIdxx() << 1;
    int index1b = index1a + 1;
    int index2a = index1a - 1;
    int index2b = index1a;
    int index1a_shared = index1a << 2;
    int index1b_shared = index1b << 2;
    int index2a_shared = index2a << 2;
    int index2b_shared = index2b << 2;

    RootbeerGpu.setSharedInteger(index1a_shared, array[index1a]);
    RootbeerGpu.setSharedInteger(index1b_shared, array[index1b]);
    //outer pass
    int arrayLength = array.length >> 1;
    for(int i = 0; i < arrayLength; ++i){
      int value1 = RootbeerGpu.getSharedInteger(index1a_shared);
      int value2 = RootbeerGpu.getSharedInteger(index1b_shared);
      int shared_value = value1;
      if(value2 < value1){
        shared_value = value2;
        RootbeerGpu.setSharedInteger(index1a_shared, value2);
        RootbeerGpu.setSharedInteger(index1b_shared, value1);
      }
      RootbeerGpu.syncthreads();
      if(index2a >= 0){
        value1 = RootbeerGpu.getSharedInteger(index2a_shared);
        //value2 = RootbeerGpu.getSharedInteger(index2b_shared);
        value2 = shared_value;
        if(value2 < value1){
          RootbeerGpu.setSharedInteger(index2a_shared, value2);
          RootbeerGpu.setSharedInteger(index2b_shared, value1);
        }
      }
      RootbeerGpu.syncthreads();
    }
    array[index1a] = RootbeerGpu.getSharedInteger(index1a_shared);
    array[index1b] = RootbeerGpu.getSharedInteger(index1b_shared);
  }
}
```


### Compiling Rootbeer Enabled Projects
1. Download the latest Rootbeer.jar from the releases
2. Program using the Kernel, Rootbeer, GpuDevice and Context class.
3. Compile your program normally with javac.
4. Pack all the classes used into a single jar using [pack](https://github.com/pcpratts/pack/)
5. Compile with Rootbeer to enable the GPU

       java -Xmx8g -jar Rootbeer.jar App-GPU-compiled.jar App-GPU.jar
       zipmerge App.jar Rootbeer.jar App-GPU-compiled.jar

6. `java -jar App.jar`f

All together: `( cd csrc && ./compile_linux_x64 ) && ant jar && ./pack-rootbeer`

### Building Rootbeer from Source

1. Clone the github repo to `rootbeer1/`
2. `cd rootbeer1/`
3. If JNI source-code was changed, then it is necessary to recompile the normally pre-compiled binaries:

       cd csrc
       ./compile_linux_x86
       ./compile_linux_x64
       ./compile_win_x86
       ./compile_win_x64
       ./compile_mac

4. `ant jar`
5. `./pack-rootbeer` (linux) or `./pack-rootbeer.bat` (windows)
6. Use the `Rootbeer.jar` (not `dist/Rootbeer1.jar`)

### Command Line Options

* `-runeasytests` = run test suite to see if things are working
* `-runtest` = run specific test case
* `-printdeviceinfo` = print out information regarding your GPU
* `-maxrregcount` = sent to CUDA compiler to limit register count
* `-noarraychecks` = remove array out of bounds checks once you get your application to work
* `-nodoubles` = you are telling rootbeer that there are no doubles and we can compile with older versions of CUDA
* `-norecursion` = you are telling rootbeer that there are no recursions and we can compile with older versions of CUDA
* `-noexceptions` = remove exception checking
* `-keepmains` = keep main methods
* `-shared-mem-size` = specify the shared memory size
* `-32bit` = compile with 32bit
* `-64bit` = compile with 64bit (if you are on a 64bit machine you will want to use just this)
* `-computecapability` = specify the Compute Capability {sm_11,sm_12,sm_20,sm_21,sm_30,sm_35} (default ALL)

Once you get started, you will find you want to use a combination of -maxregcount, -shared-mem-size and the thread count sent to the GPU to control occupancy.


### Debugging

You can use System.out.println in a limited way while on the GPU. Printing in Java requires StringBuilder support to concatenate strings/integers/etc. Rootbeer has a custom StringBuilder runtime (written with great improvements from Martin Illecker) that allows most normal printlns to work.

Since you are running on a parallel GPU, it is nice to print from a single thread

```java
public void gpuMethod(){
  if(RootbeerGpu.getThreadIdxx() == 0 && RootbeerGpu.getBlockIdxx() == 0){
    System.out.println("hello world");
  }
}
```

Once you are done debugging, you can get a performance improvement by disabling exceptions and array bounds checks (see command line options).

### Multi-GPUs (untested)

```java
List<GpuDevice> devices = rootbeer.getDevices();
GpuDevice device0 = devices.get(0);
GpuDevice device1 = devices.get(1);

Context context0 = device0.createContext(4212880);
Context context1 = device1.createContext(4212880);

context0.setCacheConfig(CacheConfig.PREFER_SHARED);
context1.setCacheConfig(CacheConfig.PREFER_SHARED);

context0.setThreadConfig(sizeBy2, outerCount, outerCount * sizeBy2);
context1.setThreadConfig(sizeBy2, outerCount, outerCount * sizeBy2);

context0.setKernel(new GPUSortKernel(array0));
context1.setKernel(new GPUSortKernel(array1));

context0.buildState();
context1.buildState();

//run using two gpus without blocking the current thread
GpuFuture future0 = context0.runAsync();
GpuFuture future1 = context1.runAsync();
future0.take();
future1.take();
```

### RootbeerGpu Builtins (compiles directly to CUDA statements)

```java
public class RootbeerGpu (){
    //returns true if on the gpu
    public static boolean isOnGpu();

    //returns blockIdx.x * blockDim.x + threadIdx.x
    public static int getThreadId();

    //returns threadIdx.x
    public static int getThreadIdxx();

    //returns blockIdx.x
    public static int getBlockIdxx();

    //returns blockDim.x
    public static int getBlockDimx();

    //returns gridDim.x;
    public static long getGridDimx();

    //__syncthreads
    public static void syncthreads();

    //__threadfence
    public static void threadfence();

    //__threadfence_block
    public static void threadfenceBlock();

    //__threadfence_system
    public static void threadfenceSystem();

    //given an object, returns the long handle
    //in GPU memory
    public static long getRef(Object obj);

    //get/set byte in shared memory. requires 1 byte.
    //index is byte offset into shared memory
    public static byte getSharedByte(int index);
    public static void setSharedByte(int index, byte value);

    //get/set char in shared memory. requires 2 bytes.
    //index is byte offset into shared memory
    public static char getSharedChar(int index);
    public static void setSharedChar(int index, char value);

    //get/set boolean in shared memory. requires 1 byte.
    //index is byte offset into shared memory
    public static boolean getSharedBoolean(int index);
    public static void setSharedBoolean(int index, boolean value);

    //get/set short in shared memory. requires 2 bytes.
    //index is byte offset into shared memory
    public static short getSharedShort(int index);
    public static void setSharedShort(int index, short value);

    //get/set integer in shared memory. requires 4 bytes.
    //index is byte offset into shared memory
    public static int getSharedInteger(int index);
    public static void setSharedInteger(int index, int value);

    //get/set long in shared memory. requires 8 bytes.
    //index is byte offset into shared memory
    public static long getSharedLong(int index);
    public static void setSharedLong(int index, long value);

    //get/set float in shared memory. requires 4 bytes.
    //index is byte offset into shared memory
    public static float getSharedFloat(int index);
    public static void setSharedFloat(int index, float value);

    //get/set double in shared memory. requires 8 bytes.
    //index is byte offset into shared memory
    public static double getSharedDouble(int index);
    public static void setSharedDouble(int index, double value);

    //atomic add value to array at index
    public static void atomicAddGlobal(int[] array, int index, int value);
    public static void atomicAddGlobal(long[] array, int index, long value);
    public static void atomicAddGlobal(float[] array, int index, float value);

    //atomic sub value from array at index
    public static void atomicSubGlobal(int[] array, int index, int value);

    //atomic exch value at index in array. old is retured
    public static int atomicExchGlobal(int[] array, int index, int value);
    public static long atomicExchGlobal(long[] array, int index, long value);
    public static float atomicExchGlobal(float[] array, int index, float value);

    //from CUDA programming guide: "reads the 32-bit word old located at the
    //address address in global memory, computes the minimum of old and val,
    //and stores the result back to memory at the same address.
    //These three operations are performed in one atomic transaction.
    //The function returns old."
    public static int atomicMinGlobal(int[] array, int index, int value);

    //from CUDA programming guide: "reads the 32-bit word old located at the
    //address address in global memory, computes the maximum of old and val,
    //and stores the result back to memory at the same address.
    //These three operations are performed in one atomic transaction.
    //The function returns old."
    public static int atomicMaxGlobal(int[] array, int index, int value);

    //from CUDA programming guide: "reads the 32-bit word old located at the
    //address address in global memory, computes (old == compare ? val : old),
    //and stores the result back to memory at the same address.
    //These three operations are performed in one atomic transaction. The function
    //returns old (Compare And Swap)."
    public static int atomicCASGlobal(int[] array, int index, int compare, int value);

    //from CUDA programming guide: "reads the 32-bit word old located at the
    //address address in global memory, computes (old & val), and stores the
    //result back to memory at the same address.
    //These three operations are performed in one atomic transaction.
    //The function returns old."
    public static int atomicAndGlobal(int[] array, int index, int value);

    //from CUDA programming guide: "reads the 32-bit word old located at the
    //address address in global memory, computes (old | val), and stores the
    //result back to memory at the same address.
    //These three operations are performed in one atomic transaction.
    //The function returns old."
    public static int atomicOrGlobal(int[] array, int index, int value);

    //from CUDA programming guide: "reads the 32-bit word old located at the
    //address address in global memory, computes (old ^ val), and stores the
    //result back to memory at the same address.
    //These three operations are performed in one atomic transaction.
    //The function returns old."
    public static int atomicXorGlobal(int[] array, int index, int value);
}
```

### Viewing Code Generation

CUDA code is generated and placed in ~/.rootbeer/generated.cu  

You can use this to find out the register / shared memory usage

    $/usr/local/cuda/bin/nvcc --ptxas-options=-v -arch sm_20 ~/.rootbeer/generated.cu

### CUDA Setup

You need to have the CUDA Toolkit and CUDA Driver installed to use Rootbeer.
Download it from http://www.nvidia.com/content/cuda/cuda-downloads.html

### License

Rootbeer is licensed under the MIT license. If you use rootbeer for any reason, please
star the repository and email me your usage and comments. I am preparing my dissertation
now.

### Examples

See [here](https://github.com/pcpratts/rootbeer1/tree/master/examples) for a variety of
examples.


### Consulting

GPU Consulting available for Rootbeer and CUDA. Please email pcpratts@trifort.org  


### Known Bugs

 - `fatal error: bits/c++config.h: No such file or directory` when running `./Rootbeer.jar -runeasytests`
   Install cross-compiling libraries:
   `sudo apt-get install gcc-4.9-multilib g++-4.9-multilib`

### Authors

Phil Pratt-Szeliga  
http://trifort.org/
Maximilian Knespel


### File Structure

Starting with the main java-file the dependency structure can be viewed with [include-spider](https://github.com/mxmlnkn/include-spider)

    incvis -q -C src src/org/trifort/rootbeer/runtime/Rootbeer.java

    Rootbeer.java
    | * Provides the Rootbeer context, which in turn provides the user API,
    | * and also GPU information and methods to start GPU calculations
    +- BlockShaper.java
    |   * used by `run` to determine the best kernel configuration
    |   * for the given workload i.e. number of started "kernels"(threads)
    +- Context.java
    |  | * Defines an abstract interface which can be implemented by e.g.
    |  | * CUDAContext.java. A more lowlevel internal API for starting
    |  | * CUDA kernels.
    |  +- CacheConfig.java
    |  | * Short Java version of cudaFuncCache enum, e.g. PREFER_SHARED
    |  +- GpuDevice.java
    |     | * A Java "struct" with getter/setter-boilerplate to hold GPU
    |     | * information. Comparable to cudaDeviceProp.
    |     | * Also contains API to create a new Context for that Device.
    |     +- CUDAContext.java
    |        | * Implements the Context interface.
    |        | * Handles the serialized class, exception and object memory.
    |        | * As well as sending the kernel to the GPU and timing the
    |        | * execution.
    |        | * Several methods are Java Native methods implemented by
    |        | * CUDARuntime.c
    |        +- ../configuration/Configuration.java
    |        |  +- ../util/ResourceReader.java
    |        |     +- ../configuration/RootbeerPaths.java
    |        +- util/Stopwatch.java
    |        +- ../runtimegpu/GpuException.java
    |        |  +- Sentinal.java
    |        +- ../generate/bytecode/Constants.java
    |        +- BufferPrinter.java
    |        |  +- Memory.java
    |        +- CheckedFixedMemory.java
    |        |  +- FixedMemory.java
    |        +- CompiledKernel.java
    |        |  +- Serializer.java
    |        +- GpuEvent.java
    |        |  +- GpuEventCommand.java
    |        |  +- GpuFuture.java
    |        |  +- Kernel.java
    |        +- StatsRow.java
    |        +- ThreadConfig.java
    +- CUDALoader.java
    +- CUDARuntime.java
    |  +- IRuntime.java
    +- OpenCLRuntime.java

    incvis -q -C src src/org/trifort/rootbeer/entry/Main.java

    generate/opencl/tweaks/GencodeOptions.java
    | * Provides Compute Capability and Architecture enums.
    | * Can generate a set of command line options to compile for
    | * all architectures that nvcc version knows
    src/org/trifort/rootbeer/util/CudaPath.java
    | * Searches for the path of nvcc or nvcc.exe
    src/org/trifort/rootbeer/util/CmdRunner.java
    | * Wrapper to correctly get the output of a command line run
    
    entry/Main.java
    +- configuration/Configuration.java
    |  +- util/ResourceReader.java
    |  |  +- configuration/RootbeerPaths.java
    +- runtime/CUDALoader.java
    |  +- runtime/Rootbeer.java
    |     +- runtime/BlockShaper.java
    |     |  +- runtime/GpuDevice.java
    |     |     +- runtime/Context.java
    |     |         +- runtime/CacheConfig.java
    |     |         +- runtime/GpuFuture.java
    |     |         |  +- runtimegpu/GpuException.java
    |     |         |     +- runtime/Sentinal.java
    |     |         +- runtime/Kernel.java
    |     |         +- runtime/StatsRow.java
    |     |            +- runtime/CUDAContext.java
    |     |               +- runtime/util/Stopwatch.java
    |     |               +- com/lmax/disruptor/EventHandler.java -> not found!
    |     |               +- com/lmax/disruptor/RingBuffer.java -> not found!
    |     |               +- com/lmax/disruptor/dsl/Disruptor.java -> not found!
    |     |               +- generate/bytecode/Constants.java
    |     |               +- runtime/BufferPrinter.java
    |     |               |  +- runtime/Memory.java
    |     |               +- runtime/CheckedFixedMemory.java
    |     |               |  +- runtime/FixedMemory.java
    |     |               |     +- org/omg/CORBA/_IDLTypeStub.java -> not found!
    |     |               +- runtime/CompiledKernel.java
    |     |               |  +- runtime/Serializer.java
    |     |               +- runtime/GpuEvent.java
    |     |               |  +- com/lmax/disruptor/EventFactory.java -> not found!
    |     |               |  +- runtime/GpuEventCommand.java
    |     |               +- runtime/ThreadConfig.java
    |     +- runtime/CUDARuntime.java
    |     |  +- runtime/IRuntime.java
    |     +- runtime/OpenCLRuntime.java
    +- entry/RootbeerCompiler.java
    |       compiler/*.java -> not found!
    |       generate/opencl/tweaks/CudaTweaks.java
    |           compressor/Compressor.java
    |               org/antlr/runtime/ANTLRStringStream.java -> not found!
    |               org/antlr/runtime/CommonTokenStream.java -> not found!
    |               org/antlr/runtime/NoViableAltException.java -> not found!
    |               org/antlr/runtime/RecognitionException.java -> not found!
    |               org/antlr/runtime/Token.java -> not found!
    |               compressor/OpenCLLexer.java
    |                   org/antlr/runtime/*.java -> not found!
    |               compressor/OpenCLParser.java
    |                   org/antlr/runtime/*.java -> not found!
    |                   org/antlr/runtime/tree/*.java -> not found!
    |           deadmethods/DeadMethods.java
    |               util/ReadFile.java
    |               deadmethods/Block.java
    |                   deadmethods/BlockParser.java
    |                       deadmethods/Segment.java
    |                           deadmethods/SegmentParser.java
    |                   deadmethods/Method.java
    |               deadmethods/LiveMethodDetector.java
    |               deadmethods/MethodAnnotator.java
    |               deadmethods/MethodNameCompressor.java
    |               deadmethods/MethodNameParser.java
    |           util/CompilerRunner.java
    |           util/CudaPath.java
    |           util/WindowsCompile.java
    |               generate/opencl/tweaks/CompileResult.java
    |               util/CmdRunner.java
    |           generate/opencl/tweaks/ParallelCompile.java
    |               runtime/BlockingQueue.java
    |               generate/opencl/tweaks/ParallelCompileJob.java
    |           generate/opencl/tweaks/Tweaks.java
    |       generate/opencl/tweaks/NativeCpuTweaks.java
    |       util/*.java -> not found!
    |       pack/Pack.java -> not found!
    |       soot/*.java -> not found!
    |       soot/options/Options.java -> not found!
    |       soot/rbclassload/DfsInfo.java -> not found!
    |       soot/rbclassload/ListClassTester.java -> not found!
    |       soot/rbclassload/ListMethodTester.java -> not found!
    |       soot/rbclassload/MethodTester.java -> not found!
    |       soot/rbclassload/RootbeerClassLoader.java -> not found!
    |       soot/util/JasminOutputStream.java -> not found!
    |       CompilerSetup.java -> not found!
    |       ForcedFields.java -> not found!
    |       KernelEntryPointDetector.java -> not found!
    |       MainTester.java -> not found!
    |       RootbeerDfs.java -> not found!
    |       TestCaseEntryPointDetector.java -> not found!
    |       TestCaseFollowTester.java -> not found!
    +- entry/RootbeerTest.java
    |       test/RootbeerTestAgent.java
    |           runtime/RootbeerGpu.java
    |           util/ForceGC.java
    |           test/ApplicationMain.java
    |               test/TestApplication.java
    |               test/TestApplicationFactory.java
    |           test/ChangeThread.java
    |               testcases/rootbeertest/gpurequired/ChangeThreadTest.java
    |                   test/TestSerialization.java
    |                   testcases/rootbeertest/gpurequired/ChangeThreadRunOnGpu.java
    |               test/TestSerializationFactory.java
    |           test/ExMain.java
    |               test/TestException.java
    |               test/TestExceptionFactory.java
    |               testcases/rootbeertest/exception/NullPointer1Test.java
    |                   testcases/rootbeertest/exception/NullPointer1RunOnGpu.java
    |               testcases/rootbeertest/exception/NullPointer2Test.java
    |                   testcases/rootbeertest/exception/NullPointer2RunOnGpu.java
    |                       testcases/rootbeertest/exception/NullPointer2Object.java
    |               testcases/rootbeertest/gpurequired/ExceptionBasicTest.java
    |                   testcases/rootbeertest/gpurequired/ExceptionBasicRunOnGpu.java
    |                       testcases/rootbeertest/gpurequired/ExceptionTestException.java
    |           test/KernelTemplateMain.java
    |               testcases/rootbeertest/kerneltemplate/DoubleToStringKernelTemplateBuilderTest.java
    |                   test/TestKernelTemplate.java
    |                   testcases/rootbeertest/kerneltemplate/DoubleToStringKernelTemplateBuilderRunOnGpu.java
    |               testcases/rootbeertest/kerneltemplate/DoubleToStringKernelTemplateTest.java
    |                   testcases/rootbeertest/kerneltemplate/DoubleToStringKernelTemplateRunOnGpu.java
    |               testcases/rootbeertest/kerneltemplate/FastMatrixTest.java
    |                   testcases/rootbeertest/kerneltemplate/MatrixKernel.java
    |               testcases/rootbeertest/kerneltemplate/GpuParametersTest.java
    |                   testcases/rootbeertest/kerneltemplate/GpuParametersRunOnGpu.java
    |               testcases/rootbeertest/kerneltemplate/GpuVectorMapTest2.java
    |                   testcases/rootbeertest/kerneltemplate/GpuVectorMap2.java
    |                       testcases/rootbeertest/kerneltemplate/GpuLongVectorPair.java
    |                   testcases/rootbeertest/kerneltemplate/GpuVectorMapRunOnGpu2.java
    |               test/TestKernelTemplateFactory.java
    |           test/LoadTestSerialization.java
    |           test/Main.java
    |               testcases/rootbeertest/SuperClass.java
    |                   testcases/rootbeertest/SuperClassRunOnGpu.java
    |                       testcases/otherpackage/CompositeClass6.java
    |                           testcases/otherpackage/CompositeClass5.java
    |                               testcases/rootbeertest/CompositeClass4.java
    |                                   testcases/rootbeertest/CompositeClass3.java
    |                                       testcases/rootbeertest/CompositeClass2.java
    |                                           testcases/rootbeertest/CompositeClass1.java
    |                                               testcases/rootbeertest/CompositeClass0.java
    |               testcases/rootbeertest/arraysum/ArraySumTest.java
    |                   testcases/rootbeertest/arraysum/ArraySum.java
    |               testcases/rootbeertest/canonical/CanonicalTest.java
    |                   testcases/rootbeertest/canonical/CanonicalKernel.java
    |                       testcases/rootbeertest/canonical2/CanonicalObject.java
    |                           testcases/rootbeertest/canonical2/CanonicalArrays.java
    |               testcases/rootbeertest/exception/NullPointer4Test.java
    |                   testcases/rootbeertest/exception/NullPointer4RunOnGpu.java
    |                       testcases/rootbeertest/exception/NullPointer4Object.java
    |               testcases/rootbeertest/gpurequired/*.java -> not found!
    |               testcases/rootbeertest/remaptest/RemapTest.java
    |                   testcases/rootbeertest/remaptest/RemapRunOnGpu.java
    |                       testcases/rootbeertest/remaptest/CallsPrivateMethod.java
    |               testcases/rootbeertest/serialization/*.java -> not found!
    |               TestSerialization.java -> not found!
    |               TestSerializationFactory.java -> not found!
    |       util/CurrJarName.java
    |       soot/G.java -> not found!
    |       soot/Modifier.java -> not found!
    |       entry/JarClassLoader.java



### Libraries / Dependencies

These libraries are distributed with this repo which bloats it's size and guarantees for at least oen copyright problem.

See folder `lib`.

  - `antlr-3.1.3.jar` [Link](http://www.antlr.org/)
     > ANTLR is an exceptionally powerful and flexible tool for parsing formal languages.
  - `asm-debug-all-5.0.3.jar` [Link](asm.ow2.org)
     > ASM is an all purpose Java bytecode manipulation and analysis framework. It can be used to modify existing classes or dynamically generate classes, directly in binary form.
  - `AXMLPrinter2.jar` [Link](https://code.google.com/archive/p/android4me/downloads)
     > Prints XML document from binary XML file
  - `commons-codec-1.6.jar` [Link](https://commons.apache.org/proper/commons-codec/)
     > Apache Commons Codec (TM) software provides implementations of common encoders and decoders such as Base64, Hex, Phonetic and URLs. 
  - `commons-collections4-4.0.jar` [Link](https://commons.apache.org/proper/commons-collections/)
     > Commons-Collections seek to build upon the JDK classes by providing new interfaces, implementations and utilities.
  - `commons-io-2.4.jar` [Link](https://commons.apache.org/proper/commons-io/)
     >
  - `dexlib2-2.0.3-dev.jar` [Link](https://github.com/JesusFreke/smali/tree/master/dexlib2)
     >
  - `disruptor-3.3.0.jar` [Link](http://lmax-exchange.github.io/disruptor/) [Github](https://github.com/LMAX-Exchange/disruptor) [Technical paper](http://lmax-exchange.github.com/disruptor/files/Disruptor-1.0.pdf)
     > A High Performance Inter-Thread Messaging Library
  - `guava-18.0.jar` [Link]()
     >
  - `hamcrest-all-1.3.jar` [Link](http://hamcrest.org/JavaHamcrest/)
     > Hamcrest is a framework for writing matcher objects allowing 'match' rules to be defined declaratively.
  - `jasminclasses-2.5.0.jar` [Link]()
     >
  - `pack.jar` [Link](https://github.com/pcpratts/pack)
     > merges jars to one fat jar (same as zipmerge) plus it removes the line `Class-Path:.*` from `META-INF/MANIFEST.MF` from the main jar (would be the last jar witht he zipmerge command).
  - `polyglotclasses-1.3.5.jar` [Link]()
     >
  - `sootclasses-rbclassload.jar` [Link]()
     >
  - `util-2.0.3-dev.jar` [Link]()
     >


### Developing

Some classes do have main functions for testing purposes. Start them e.g. with

    java -classpath build/classes/ org.trifort.rootbeer.runtime.BlockShaper
    
Note that the second argument may not be different because of the `package` keyword at the top of this file. Instead adjust the classpath if necessary!


### ToDo 
    
   - add compiler warnings
   - streamlining pack-rootbeer
   - add support for #  define with spaces
   - reduce debug output and explain it (now I udnerstand what block and segments mean)
   - add support for /* */ style comments -> TYPE_COMMENT recognition somehow
   - check which metrics do change on a changed kernel, e.g. when adding a local variable

