/**
 * Caused by: java.lang.StringIndexOutOfBoundsException: String index out of range: -2
 * 	at java.lang.String.substring(String.java:1904)
 * 	at org.trifort.rootbeer.deadmethods.MethodNameParser.parseMethodName(MethodNameParser.java:55)
 * 	at org.trifort.rootbeer.deadmethods.MethodNameParser.parse(MethodNameParser.java:22)
 * 	at org.trifort.rootbeer.deadmethods.DeadMethods.parseString(DeadMethods.java:49)
 * 	at org.trifort.rootbeer.generate.opencl.tweaks.CudaTweaks.compileProgram(CudaTweaks.java:95)
 * 	... 9 more
 *
 * struct m_Local {
 *     unsigned long long dpObjectMem,        // 0
 *     unsigned long long objectMemSizeDiv16, // 1
 *     unsigned long long dpClassMem          // 2
 * }
 * @see CudaHeader.c csrc/org/trifort/rootbeer/runtime/CUDAContext.c
 */

__device__ int
org_trifort_classConstant( int type_num )
{
    int * temp = (int *) m_Local[2]; // dpClassMem
    return temp[type_num];
}

__device__  char *
org_trifort_gc_deref( int compressedAddress )
{
    char * data_arr = (char *) m_Local[0]; // dpObjectMem
    return &data_arr[ ( (long long) compressedAddress ) << MallocAlignZeroBits ];
}


/**
 *TODO: don't pass gc_info everywhere because free pointer is __device__
 *
 * This gets initialized to object_mem.get_heap_end() / 16
 * Meaning it is not directly a pointer, but a software "pointer" which
 * will be added to a real pointer.
 * The elements coming before it are the static data members of the kernel.
 * It is in units of 16 Bytes!
 */
__device__ int global_free_pointer;

int __device__ padNumberTo
(
    int const number,
    int const multiplier
)
{
    int const padded = number % multiplier == 0 ? number :
                       number + multiplier - number % multiplier;
    assert( padded % multiplier == 0 );
    assert( padded >= number );
    assert( padded <  number + multiplier );
    return padded;
}

/**
 * Allocate at least the specified size in bytes.
 * Align to MallocAlignBytes (16) Bytes.
 *
 * gc means garbage collector.
 *
 * @return address for allocated memory chunk
 */
__device__ int
org_trifort_gc_malloc_no_fail( int size )
{
    int const lastAddress = atomicAdd( &global_free_pointer,
                                       padNumberTo( size, MallocAlignBytes ) / MallocAlignBytes );
    return lastAddress;
}

/**
 * Identical to org_trifort_gc_malloc_no_fail, but checks if the returned
 * address is actually valid!
 */
__device__ int
org_trifort_gc_malloc( int size )
{
    int const & objectMemSizeDiv16 = (int) m_Local[1];
    int const address = org_trifort_gc_malloc_no_fail( size );
    /**
     * int const end = address + size/16 + 1; // this looks very wrong!
     * if ( end >= objectMemSizeDiv16 )
     *                           0 <= size <= 15 -> 1
     * This is floor( size/16 ) + 1. It is stricter than this new version, so
     * it shouldn't have led to a segfault, it was just confusing and lax
     */
    if ( address + padNumberTo( size, MallocAlignBytes ) / MallocAlignBytes > objectMemSizeDiv16 )
    {
        assert( false && "Garbage collector ran out of memory!" );
        // this needs to be checked by the caller! Else a segfault before the
        // allocated memory might occur ... */
        return -1;
    }
    return address;
}

__device__
long long java_lang_System_nanoTime(int * exception){
    return (long long) clock64();
}

__global__ void entry
(
    int * handles   ,
    int * exceptions,
    int   numThreads,
    int   usingKernelTemplates
)
{
int totalThreadId =
getThreadId();
if ( totalThreadId < numThreads )
{
int exception = 0;
int handle;
if ( usingKernelTemplates )
handle = handles[0];
else
handle = handles[totalThreadId];
%%invoke_run%%(handle, &exception);
if ( %%using_exceptions%% )
exceptions[totalThreadId] = exception;
}
}
