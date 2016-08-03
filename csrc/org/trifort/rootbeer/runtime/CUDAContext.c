
#include "CUDAContext.h"

#include <cuda.h>
#include <stdio.h>      // printf, sprintf
#include <stdlib.h>     // malloc, free
#include <string.h>     // strlen
#include <assert.h>

#include "PointerCasting.h"
#include "CUDARuntime.h"
#include "Stopwatch.h"


#define DEBUG_CUDA_CONTEXT 0

/* e.g. because of alignment to 16 the last 4 bits will always be 0
 * and can be cut off for compression! */
#define N_ALIGNED_ZERO_BITS 4


#define CE( STATUS )                                                \
{                                                                   \
    const CUresult status = STATUS;                                 \
    const char *errorName, *errorString;                            \
    cuGetErrorName  ( status, &errorName   );                       \
    cuGetErrorString( status, &errorString );                       \
    if ( status != CUDA_SUCCESS )                                   \
    {                                                               \
        char msg[8*1024];                                           \
        sprintf( msg,                                               \
            "Line %i, command: "#STATUS"\n failed with "            \
            "error code %i (%.1000s) : %.4000s\n",                  \
            __LINE__, status, errorString, errorString );           \
        throw_cuda_error_exception( env, msg, status, s->device );  \
        return;                                                     \
    }                                                               \
}

#define __PRINTOUT(...) sprintf( &output[ strlen( output ) ],  __VA_ARGS__ );
//#define __PRINTOUT(...) printf( __VA_ARGS__ );


/* One reason for using the CUDA driver api is that nvcc isn't needed,
 * instead this can be compiled with gcc! */
struct ContextState
{
    CUdevice           device                       ;
    CUcontext          context                      ;
    CUmodule           module                       ;
    CUfunction         function                     ;

    CUdeviceptr        gpu_info_space               ;
    CUdeviceptr        gpu_object_mem               ;
    CUdeviceptr        gpu_handles_mem              ;
    CUdeviceptr        gpu_exceptions_mem           ;
    CUdeviceptr        gpu_class_mem                ;
    CUdeviceptr        gpu_heap_end                 ;

    void             * cpu_object_mem               ;
    void             * cpu_handles_mem              ;
    void             * cpu_exceptions_mem           ;
    void             * cpu_class_mem                ;

    jlong              cpu_object_mem_size          ;
    jlong              cpu_handles_mem_size         ;
    jlong              cpu_exceptions_mem_size      ;
    jlong              cpu_class_mem_size           ;

    jint             * info_space                   ;
    jint               block_count_x                ;
    jint               block_count_y                ;
    jint               using_kernel_templates_offset;
    jint               using_exceptions             ;
    jint               context_built                ;

    struct stopwatch   execMemcopyToDevice          ;
    struct stopwatch   execGpuRun                   ;
    struct stopwatch   execMemcopyFromDevice        ;
};

/* interfaced java classes and methods which we want to access / call */
jclass    cuda_memory_class  ;
jmethodID get_address_method ;
jmethodID get_size_method    ;
jmethodID get_heap_end_method;
jmethodID set_heap_end_method;
jclass    stats_row_class    ;
jmethodID set_driver_times   ;


/**
* Throws a runtimeexception called CudaMemoryException
* allocd - number of bytes tried to allocate
* id - variable the memory assignment was for
*/
void throw_cuda_error_exception
(
    JNIEnv     * env    ,
    const char * message,
    int          error  ,
    CUdevice     device
)
{
    char     msg[4*1024];
    jclass   exp;
    jfieldID fid;
    int      a = 0;
    int      b = 0;
    char     name[1024];

    exp = (*env)->FindClass(env,"org/trifort/rootbeer/runtime/CudaErrorException");

    // we truncate the message to 900 characters to stop any buffer overflow
    switch ( error )
    {
        case CUDA_ERROR_OUT_OF_MEMORY:
        {
            sprintf( msg, "CUDA_ERROR_OUT_OF_MEMORY: %.900s",message );
            break;
        }
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
        {
            cuDeviceGetName(name,1024,device);
            cuDeviceComputeCapability(&a, &b, device);
            sprintf( msg, "No binary for gpu. %.900s Selected %s (%d.%d). 2.0 compatibility required.", message, name, a, b );
            break;
        }
        default:
            sprintf( msg, "ERROR STATUS:%i : %.900s", error, message );
    }

    fid = (*env)->GetFieldID(env,exp, "cudaError_enum", "I");
    (*env)->SetLongField(env,exp,fid, (jint)error);
    (*env)->ThrowNew(env,exp,msg);
    return;
}

/**
 * Cache function pointers to Java class methods
 */
JNIEXPORT void JNICALL
Java_org_trifort_rootbeer_runtime_CUDAContext_initializeDriver
( JNIEnv *env, jobject this_ref )
{
    cuda_memory_class   = (*env)->FindClass  ( env, "org/trifort/rootbeer/runtime/FixedMemory"   );
    get_address_method  = (*env)->GetMethodID( env, cuda_memory_class, "getAddress"   , "()J"    );
    get_size_method     = (*env)->GetMethodID( env, cuda_memory_class, "getSize"      , "()J"    );
    get_heap_end_method = (*env)->GetMethodID( env, cuda_memory_class, "getHeapEndPtr", "()J"    );
    set_heap_end_method = (*env)->GetMethodID( env, cuda_memory_class, "setHeapEndPtr", "(J)V"   );
    stats_row_class     = (*env)->FindClass  ( env, "org/trifort/rootbeer/runtime/StatsRow"      );
    set_driver_times    = (*env)->GetMethodID( env, stats_row_class, "setDriverTimes" , "(JJJ)V" );
}

JNIEXPORT jlong JNICALL Java_org_trifort_rootbeer_runtime_CUDAContext_allocateNativeContext
  (JNIEnv *env, jobject this_ref)
{
    struct ContextState * ret = (struct ContextState *) malloc(sizeof(struct ContextState));
    ret->context_built = 0;
    return pointerToJlong( ret );
}

JNIEXPORT void JNICALL
Java_org_trifort_rootbeer_runtime_CUDAContext_freeNativeContext
( JNIEnv *env, jobject this_ref, jlong reference )
{
    struct ContextState * s /* stateObject */ = (struct ContextState *) jlongToPointer( reference );
    if ( s->context_built )
    {
        free        ( s->info_space         );
        cuMemFree   ( s->gpu_info_space     );
        cuMemFree   ( s->gpu_object_mem     );
        cuMemFree   ( s->gpu_handles_mem    );
        cuMemFree   ( s->gpu_exceptions_mem );
        cuMemFree   ( s->gpu_class_mem      );
        cuMemFree   ( s->gpu_heap_end       );
        cuCtxDestroy( s->context            );
    }
    free( s );
}


/**
 * Sets the state as specified by the multitude of parameters
 *
 * Sets specified GPU device, creates CUDA context, sets shared memory,
 * configuration, kernel parameters and kernel configuration.
 *
 * @todo Can on thread have multiple CUDA contexts and work on multiple
 * GPU devices in parallel? Or do I need to call cuDeviceSet before each
 * command?
 **/
JNIEXPORT void JNICALL Java_org_trifort_rootbeer_runtime_CUDAContext_nativeBuildState
(
    JNIEnv *   env             ,
    jobject    rThisRef        ,
    jlong      rNativeContext  ,
    jint       rDeviceIndex    ,
    jbyteArray rCubinFile      ,
    jint       rCubinLength    ,
    jint       rThreadCountX   ,
    jint       rThreadCountY   ,
    jint       rThreadCountZ   ,
    jint       rBlockCountX    ,
    jint       rBlockCountY    ,
    jint       rNumThreads     ,
    jobject    rObjectMem      , /* instances of Memory.java */
    jobject    rHandlesMem     , /* either:                  */
    jobject    rExceptionsMem  , /*     FixedMemory.java     */
    jobject    rClassMem       , /* or: CheckedMemody.java   */
    jint       rUsingExceptions,
    jint       rCacheConfig
)
{
    /* just basically an alias + pointer conversion */
    struct ContextState * const s /* stateObject */ = (struct ContextState *) jlongToPointer( rNativeContext );

    s->block_count_x    = rBlockCountX    ;
    s->block_count_y    = rBlockCountY    ;
    s->using_exceptions = rUsingExceptions;

    CE( cuDeviceGet( &(s->device) , rDeviceIndex ) )
    CE( cuCtxCreate( &(s->context), CU_CTX_MAP_HOST, s->device ) )

    /* Loads fatCubin (device code for multiple architectures) into a module */
    void * fatcubin = malloc( rCubinLength ); // holds cubin in memory
    /* http://docs.oracle.com/javase/1.5.0/docs/guide/jni/spec/functions.html#wp1716
     * >copies< C javaByteArray rCubinFile into raw C data fatcubin */
    (*env)->GetByteArrayRegion( env, rCubinFile, 0, rCubinLength, fatcubin );
    /* http://docs.nvidia.com/cuda/cuda-c-programming-guide/#module */
    CE( cuModuleLoadFatBinary( &(s->module), fatcubin) )
    free( fatcubin );
    /* get kernel which to start from loaded module (i.e. ptx library) */
    CE( cuModuleGetFunction( &(s->function), s->module, "_Z5entryPiS_ii" ) )
    /* c++filt '_Z5entryPiS_ii' -> entry(int*, int*, int, int)
     * As can be seen in generated_unix.xu the meaning of those kernel
     * parameters is:
     *    __global__ void entry
     *    (
     *        int * handles,
     *        int * exceptions,
     *        int numThreads,
     *        int usingKernelTemplates
     *    )
     */

    CUfunc_cache cache_config_enum; // prefer shared, L1, ...
    if ( rCacheConfig != 0 )
    {
        switch ( rCacheConfig )
        {
            case 1: cache_config_enum = CU_FUNC_CACHE_PREFER_SHARED; break;
            case 2: cache_config_enum = CU_FUNC_CACHE_PREFER_L1    ; break;
            case 3: cache_config_enum = CU_FUNC_CACHE_PREFER_EQUAL ; break;
        }
        CE( cuFuncSetCacheConfig( s->function, cache_config_enum ) )
    }

    s->cpu_object_mem     = jlongToPointer( (*env)->CallLongMethod( env, rObjectMem    , get_address_method ) );
    s->cpu_handles_mem    = jlongToPointer( (*env)->CallLongMethod( env, rHandlesMem   , get_address_method ) );
    s->cpu_exceptions_mem = jlongToPointer( (*env)->CallLongMethod( env, rExceptionsMem, get_address_method ) );
    s->cpu_class_mem      = jlongToPointer( (*env)->CallLongMethod( env, rClassMem     , get_address_method ) );

    s->cpu_object_mem_size     = (*env)->CallLongMethod( env, rObjectMem    , get_size_method );
    s->cpu_handles_mem_size    = (*env)->CallLongMethod( env, rHandlesMem   , get_size_method );
    s->cpu_exceptions_mem_size = (*env)->CallLongMethod( env, rExceptionsMem, get_size_method );
    s->cpu_class_mem_size      = (*env)->CallLongMethod( env, rClassMem     , get_size_method );

    s->info_space = (jint *) malloc( sizeof( *(s->info_space) ) );

    /** allocate corresponding memory on gpu **/
    CE( cuMemAlloc( &( s->gpu_info_space  ), sizeof( *(s->info_space) ) ) )
    CE( cuMemAlloc( &( s->gpu_object_mem  ), s->cpu_object_mem_size     ) )
    CE( cuMemAlloc( &( s->gpu_handles_mem ), s->cpu_handles_mem_size    ) )
    CE( cuMemAlloc( &( s->gpu_class_mem   ), s->cpu_class_mem_size      ) )
    CE( cuMemAlloc( &( s->gpu_heap_end    ), sizeof( jint )             ) )
    if ( s->using_exceptions )
        CE( cuMemAlloc( &( s->gpu_exceptions_mem ), s->cpu_exceptions_mem_size ) )
    else
        s->gpu_exceptions_mem = (CUdeviceptr) 0;

    /******** set function parameters ********/
    /* cuParamSet is officially deprecated ...
     * http://horacio9573.no-ip.org/cuda/group__CUDA__EXEC__DEPRECATED_gdf689dac0db8f6c1232c339d3f923554.html
     * Since CUDA Driver 4.0 i.e. NVCC/CUDA 4.0 there is cuLaunchKernel instead
     * http://stackoverflow.com/questions/19240658/cuda-kernel-launch-parameters-explained-right
     * It seems like Rootbeer wants to support even 3.0 and 3.2, see
     * src/org/trifort/rootbeer/generate/opencl/tweaks/GencodeOptions.java
     */
    /* Sets the total size in bytes needed by the function parameters of the kernel */
    int const nBytesTotalParameters = 2 * sizeof(CUdeviceptr) + 2 * sizeof(int);
    CE( cuParamSetSize(s->function, nBytesTotalParameters ) )
    int offset = 0; // parameter list offset for cuParamSet{i,v}
    #define setNextParam( SRC, NBYTES )                                 \
        CE( cuParamSetv( s->function, offset, (void *) SRC, NBYTES ) )  \
        offset += NBYTES;
    setNextParam( &(s->gpu_handles_mem   ), sizeof(CUdeviceptr) )
    setNextParam( &(s->gpu_exceptions_mem), sizeof(CUdeviceptr) )
    CE( cuParamSeti( s->function, offset, rNumThreads) ); offset += sizeof(int);
    /* The last kernel parameter is set by cudaRun, remember only the offset! */
    assert( offset >= 0 );
    assert( (unsigned int) offset == nBytesTotalParameters - sizeof(int) );
    s->using_kernel_templates_offset = offset;

    CE( cuFuncSetBlockShape( s->function, rThreadCountX,
                                          rThreadCountY,
                                          rThreadCountZ ) )
    s->context_built = 1;

    /* debug output, trying to understand rootbeer */
    #if ( ! defined( NDEBUG ) ) && ( DEBUG_CUDA_CONTEXT >= 10 )
        char * output = malloc( 1024*1024 );
        output[0] = '\0';

        __PRINTOUT( "+-------------- [nativeBuildState] --------------\n" );
        #define __PRINTI( NAME ) __PRINTOUT( "| %32s = %10i\n", #NAME, NAME );
        __PRINTI( rCubinLength  )
        __PRINTI( rThreadCountX )
        __PRINTI( rThreadCountY )
        __PRINTI( rThreadCountZ )
        __PRINTI( rBlockCountX  )
        __PRINTI( rBlockCountY  )
        __PRINTI( rNumThreads   )
        __PRINTI( rCacheConfig  )
        __PRINTI( rDeviceIndex  )
        __PRINTI( s->using_kernel_templates_offset )
        __PRINTOUT( "|\n" );

        #define __PRINTP( PTR, SIZE )                                   \
            __PRINTOUT( "| %32s = %p (size: %10lu B = %10lu KiB)\n",    \
                        #PTR, PTR, SIZE, SIZE / 1024 );

        __PRINTP( (void*) s->cpu_object_mem    , (unsigned long) s->cpu_object_mem_size     )
        __PRINTP( (void*) s->cpu_handles_mem   , (unsigned long) s->cpu_handles_mem_size    )
        __PRINTP( (void*) s->cpu_exceptions_mem, (unsigned long) s->cpu_exceptions_mem_size )
        __PRINTP( (void*) s->cpu_class_mem     , (unsigned long) s->cpu_class_mem_size      )
        __PRINTOUT( "|\n" );

        __PRINTP( (void*) s->gpu_info_space    , sizeof( *(s->info_space) )              )
        __PRINTP( (void*) s->gpu_object_mem    , (unsigned long) s->cpu_object_mem_size  )
        __PRINTP( (void*) s->gpu_handles_mem   , (unsigned long) s->cpu_handles_mem_size )
        __PRINTP( (void*) s->gpu_class_mem     , (unsigned long) s->cpu_class_mem_size   )
        __PRINTP( (void*) s->gpu_heap_end      , sizeof( jint )                          )
        __PRINTP( (void*) s->gpu_exceptions_mem, sizeof( jint )                          )
        __PRINTOUT( "|\n" );

        printf( "%s", output );
        free( output );

        #undef __PRINTI
        #undef __PRINTP
    #endif
}

JNIEXPORT void JNICALL Java_org_trifort_rootbeer_runtime_CUDAContext_cudaRun
(
    JNIEnv * env,
    jobject  this_ref,
    jlong    nativeContext,
    jobject  object_mem,    /**< maybe better to save object in member on native build state, not just address? -> wouldn't update member variables -> address to object then? */
    jint     using_kernel_templates,
    jobject  stats_row
)
{
    struct ContextState * const s = (struct ContextState *) jlongToPointer( nativeContext );

    stopwatchStart(&(s->execMemcopyToDevice));

    jlong heap_end_long;
    heap_end_long = (*env)->CallLongMethod( env, object_mem, get_heap_end_method );
    #if ( ! defined( NDEBUG ) ) && ( DEBUG_CUDA_CONTEXT >= 10 )
        unsigned int nMaxBytesOutput = 16*1024*1024;
        char * output = malloc( nMaxBytesOutput );
        output[0] = '\0';

        __PRINTOUT( "[cudaRun] heap_end_long = %lu\n", heap_end_long );
    #endif
    assert( heap_end_long >= 0 ); /* it also may not be -1 which is sometimes used for null ! */
    jint heap_end_int = (jint)( heap_end_long >> N_ALIGNED_ZERO_BITS );
    /* check that compression is reversible (includes check for numbers not fitting into jint) */
    assert( ( (jlong) heap_end_int ) << N_ALIGNED_ZERO_BITS == heap_end_long );
    s->info_space[0] = heap_end_int;

    unsigned long long hostMLocal[3];
    hostMLocal[0] = s->gpu_object_mem;
    assert( s->cpu_object_mem_size >= 0 );
    hostMLocal[1] = s->cpu_object_mem_size >> N_ALIGNED_ZERO_BITS;
    assert( s->cpu_object_mem_size >= 0 );
    assert( sizeof(jint) == 4 &&
            "If this assert fails the assert after this may need to be rewritten!" );
    assert( hostMLocal[1] < 2147483647 );
    /* equality would be more clean programming, but it just isn't, and if
     * the hostMLocal version is smaller no harm is done, as that size is used
     * for the rootbeer garbage collector. (Currently the size is set manually
     * by the user in createContext arguments or with */
    assert( hostMLocal[1] << N_ALIGNED_ZERO_BITS <= (unsigned long long) s->cpu_object_mem_size );
    hostMLocal[2] = s->gpu_class_mem;

    /* Get address and size of global_free_pointer which is defined in
     * src/org/trifort/rootbeer/generate/opencl/CudaKernel.c
     * from the loaded module i.e. ptx library-like binary. */
    size_t nBytesDeviceGlobalFreePointer = 0;
    CUdeviceptr deviceGlobalFreePointer;
    CE( cuModuleGetGlobal( &deviceGlobalFreePointer,
                           &nBytesDeviceGlobalFreePointer,
                           s->module,
                           "global_free_pointer" ) )
    assert( nBytesDeviceGlobalFreePointer == sizeof( *(s->info_space) ) );

    CUdeviceptr deviceMLocal;
    size_t nBytesDeviceMLocal = 0;
    CE( cuModuleGetGlobal( &deviceMLocal, &nBytesDeviceMLocal, s->module,
                           "m_Local" ) )
    assert( nBytesDeviceMLocal == sizeof( hostMLocal ) );

    /* debug output, trying to understand rootbeer */
    #if ( ! defined( NDEBUG ) ) && ( DEBUG_CUDA_CONTEXT >= 10 )
        #define __PRINTI( NAME ) \
            __PRINTOUT( "| %32s = % 10i\n", #NAME, NAME );
        __PRINTOUT( "+-------------- [nativeBuildState] --------------\n" );
        __PRINTI( heap_end_int )

        #define __PRINTP( PTR, SIZE )                                   \
            __PRINTOUT( "| %32s = %p (size: %10lu B = %10lu KiB)\n",    \
                        #PTR, PTR, SIZE, SIZE / 1024 );

        __PRINTP( (void*) deviceGlobalFreePointer, nBytesDeviceGlobalFreePointer )
        __PRINTP( (void*) deviceMLocal           , nBytesDeviceMLocal            )
        __PRINTOUT( "|\n" );

        //__PRINTOUT( "+-------- Handles to send to GPU:\n" ):
        //int i = 0;
        //while ( (i+1) * sizeof(long) <= s->cpu_handles_mem_size )
        //{
        //    if ( i % 8 == 0 )
        //        __PRINTOUT( "\n" );
        //
        //    __PRINTOUT( "% 8d ", ((long*)s->cpu_handles_mem)[i] );
        //    ++i;
        //}

        /* make backup of handle memory */
        void * handlesSentToGpu = malloc( s->cpu_handles_mem_size );
        memcpy( handlesSentToGpu, s->cpu_handles_mem, s->cpu_handles_mem_size );
        void * objectsSentToGpu = malloc( s->cpu_object_mem_size );
        memcpy( objectsSentToGpu, s->cpu_object_mem, s->cpu_object_mem_size  );

        #undef __PRINTI
        #undef __PRINTP
    #endif

    /** copy data **/

    /* why not gpu_info_space used here ??? gpu_info_space is unused else.
     * also info_space holds gpu_heap_end >> 4. Why this duplication? */
    CE( cuMemcpyHtoD( deviceGlobalFreePointer, s->info_space     , sizeof( *(s->info_space) ) ) )
    CE( cuMemcpyHtoD( deviceMLocal           , hostMLocal        , sizeof( hostMLocal )       ) )
    CE( cuMemcpyHtoD( s->gpu_object_mem      , s->cpu_object_mem , s->cpu_object_mem_size     ) )
    CE( cuMemcpyHtoD( s->gpu_handles_mem     , s->cpu_handles_mem, s->cpu_handles_mem_size    ) )
    CE( cuMemcpyHtoD( s->gpu_class_mem       , s->cpu_class_mem  , s->cpu_class_mem_size      ) )
    CE( cuMemcpyHtoD( s->gpu_heap_end        , &(heap_end_int)   , sizeof( heap_end_int )     ) )
    if ( s->using_exceptions )
        CE( cuMemcpyHtoD( s->gpu_exceptions_mem, s->cpu_exceptions_mem, s->cpu_exceptions_mem_size ) )

    stopwatchStop( &s->execMemcopyToDevice );

    /** launch **/
    stopwatchStart( &s->execGpuRun );

    CE( cuParamSeti ( s->function, s->using_kernel_templates_offset, using_kernel_templates ) )
    CE( cuLaunchGrid( s->function, s->block_count_x, s->block_count_y) )
    CE( cuCtxSynchronize() )

    stopwatchStop(&(s->execGpuRun));

    /** copy data back **/
    stopwatchStart(&(s->execMemcopyFromDevice));

    CE( cuMemcpyDtoH( s->info_space, deviceGlobalFreePointer, sizeof( *(s->info_space) ) ) )
    heap_end_long = s->info_space[0];
    /**
     * An old bug was using * 16 instead of *= 16 (I think some compiler
     * warning might have had noticed this...) resulting in:
     *   1324514 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   1323744 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   1323263 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   1322902 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   [...]
     *   0 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   0 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   0 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     *   0 inside 0 outside
     *     => Something is wrong! Don't add up to 1684701
     * This means the nHitsB array and more than half of nHitsA array were
     * not copied back to host from gpu resulting in those values being 0.
     * This means that nHitsB lies after nHistA in the the rootbeer managed
     * heap.
     */
    heap_end_long <<= N_ALIGNED_ZERO_BITS;
    CE( cuMemcpyDtoH( s->cpu_object_mem, s->gpu_object_mem, heap_end_long ) )
    if ( s->using_exceptions )
        CE( cuMemcpyDtoH(s->cpu_exceptions_mem, s->gpu_exceptions_mem, s->cpu_exceptions_mem_size) )
    (*env)->CallVoidMethod( env, object_mem, set_heap_end_method, heap_end_long );

    stopwatchStop(&(s->execMemcopyFromDevice));

    /* save performance statistics to Java readable class */
    (*env)->CallVoidMethod( env, stats_row, set_driver_times,
        stopwatchTimeMS( &(s->execMemcopyToDevice   ) ),
        stopwatchTimeMS( &(s->execGpuRun            ) ),
        stopwatchTimeMS( &(s->execMemcopyFromDevice ) )
    );

    #if ( ! defined( NDEBUG ) ) && ( DEBUG_CUDA_CONTEXT >= 10 )
        {
            __PRINTOUT( "+-------- Handles after GPU call ('!' means they changed !):" );
            int i = 0;
            while ( (i+1) * sizeof(jint) <= (unsigned long) s->cpu_handles_mem_size )
            {
                if ( i % 8 == 0 )
                    __PRINTOUT( "\n%6i : ", i );
                if ( ((jint*)s->cpu_handles_mem)[i] == ((jint*)handlesSentToGpu)[i] )
                {
                    __PRINTOUT( "%6i ", ((jint*)s->cpu_handles_mem)[i] );
                }
                else
                {
                    __PRINTOUT( "!%i(%i) ", ((jint*)s->cpu_handles_mem)[i],
                                            ((jint*)handlesSentToGpu  )[i] );
                }
                ++i;
            }
            __PRINTOUT( "\n\n" );
        }

        for ( long i = 0; i < s->cpu_handles_mem_size; ++i )
        {
            assert( ( (char*) s->cpu_handles_mem )[i] ==
                    ( (char*) handlesSentToGpu   )[i] &&
                    "[CUDAContext.c] handles memory changed indicating a problem!" );
        }
        free( handlesSentToGpu );

        {
            __PRINTOUT( "+-------- Object Memory after GPU call ('!' means they changed):" );
            int i = 0;
            while ( (i+1) * sizeof(jint) <= (unsigned long) s->cpu_object_mem_size )
            {
                /* also print the last 1024 jints */
                if ( i == 1024 && s->cpu_object_mem_size / sizeof(jint) > 2048 )
                {
                    i = s->cpu_object_mem_size / sizeof(jint) - 1024;
                    __PRINTOUT( "\n ... \n" );
                    continue;
                }

                if ( i % 8 == 0 )
                    __PRINTOUT( "\n%5i : ", i );
                if ( ((jint*)s->cpu_object_mem)[i] == ((jint*)objectsSentToGpu)[i] )
                {
                    __PRINTOUT( "%6i ", ((jint*)s->cpu_object_mem)[i] );
                }
                else
                {
                    __PRINTOUT( "!%i(%i) ", ((jint*)s->cpu_object_mem)[i],
                                            ((jint*)objectsSentToGpu )[i] );
                }
                ++i;
            }
            __PRINTOUT( "\n\n" );
        }
        free( objectsSentToGpu );

        printf( "%s", output );
        free( output );
    #endif
}

#undef CE
