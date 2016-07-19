#include "CUDARuntime.h"
#include "Stopwatch.h"
#include <cuda.h>
#include <stdio.h>      // printf, sprintf
#include <stdlib.h>     // malloc, free
#include <stddef.h>     // NULL
#include <assert.h>

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
            sprintf(msg, "CUDA_ERROR_OUT_OF_MEMORY: %.900s",message);
            break;
        }
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
        {
            cuDeviceGetName(name,1024,device);
            cuDeviceComputeCapability(&a, &b, device);
            sprintf(msg, "No binary for gpu. %.900s Selected %s (%d.%d). 2.0 compatibility required.", message, name, a, b);
            break;
        }
        default:
            sprintf(msg, "ERROR STATUS:%i : %.900s", error, message);
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
    return (jlong) ret;
}

JNIEXPORT void JNICALL
Java_org_trifort_rootbeer_runtime_CUDAContext_freeNativeContext
( JNIEnv *env, jobject this_ref, jlong reference )
{
    struct ContextState * s /* stateObject */ = (struct ContextState *) reference;
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
    struct ContextState * const s /* stateObject */ = (struct ContextState *) rNativeContext;

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

    s->cpu_object_mem     = (void *) (*env)->CallLongMethod( env, rObjectMem    , get_address_method );
    s->cpu_handles_mem    = (void *) (*env)->CallLongMethod( env, rHandlesMem   , get_address_method );
    s->cpu_exceptions_mem = (void *) (*env)->CallLongMethod( env, rExceptionsMem, get_address_method );
    s->cpu_class_mem      = (void *) (*env)->CallLongMethod( env, rClassMem     , get_address_method );

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
    assert( offset == nBytesTotalParameters - sizeof(int) );
    s->using_kernel_templates_offset = offset;

    CE( cuFuncSetBlockShape( s->function, rThreadCountX,
                                          rThreadCountY,
                                          rThreadCountZ ) )
    s->context_built = 1;

    /* debug output, trying to understand rootbeer */
    #ifndef NDEBUG
        #define __PRINTI( NAME ) \
            printf( "| %32s = % 10i\n", #NAME, NAME );
        printf( "+-------------- [nativeBuildState] --------------\n" );
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
        printf( "|\n" );

        #define __PRINTP( PTR, SIZE ) \
            printf( "| %32s = %p (size: % 10lu B = % 10i KiB)\n", #PTR, PTR, SIZE, SIZE / 1024 );

        __PRINTP( s->cpu_object_mem    , s->cpu_object_mem_size     )
        __PRINTP( s->cpu_handles_mem   , s->cpu_handles_mem_size    )
        __PRINTP( s->cpu_exceptions_mem, s->cpu_exceptions_mem_size )
        __PRINTP( s->cpu_class_mem     , s->cpu_class_mem_size      )
        printf( "|\n" );

        __PRINTP( s->gpu_info_space    , sizeof( *(s->info_space) ) )
        __PRINTP( s->gpu_object_mem    , s->cpu_object_mem_size     )
        __PRINTP( s->gpu_handles_mem   , s->cpu_handles_mem_size    )
        __PRINTP( s->gpu_class_mem     , s->cpu_class_mem_size      )
        __PRINTP( s->gpu_heap_end      , sizeof( jint )             )
        __PRINTP( s->gpu_exceptions_mem, sizeof( jint )             )
        printf( "|\n" );

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
    CUdeviceptr deviceGlobalFreePointer;
    struct ContextState * const s = (struct ContextState *) nativeContext;

    stopwatchStart(&(s->execMemcopyToDevice));

    jlong heap_end_long;
    heap_end_long = (*env)->CallLongMethod(env, object_mem, get_heap_end_method);
    heap_end_long >>= 4;
    jint heap_end_int = (jint) heap_end_long;
    s->info_space[0] = heap_end_int;

    size_t bytes;
    CE( cuModuleGetGlobal(&deviceGlobalFreePointer, &bytes, s->module, "global_free_pointer") )
    CUdeviceptr deviceMLocal;
    CE( cuModuleGetGlobal(&deviceMLocal, &bytes, s->module, "m_Local") )

    /** copy data **/
    unsigned long long hostMLocal[3];
    hostMLocal[0] = s->gpu_object_mem;
    hostMLocal[1] = s->cpu_object_mem_size >> 4;    /* WHAT is up with this bitshift ? */
    hostMLocal[2] = s->gpu_class_mem;

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

    stopwatchStop(&(s->execMemcopyToDevice));

    /** launch **/
    stopwatchStart(&(s->execGpuRun));

    CE( cuParamSeti ( s->function, s->using_kernel_templates_offset, using_kernel_templates) )
    CE( cuLaunchGrid( s->function, s->block_count_x, s->block_count_y) )
    CE( cuCtxSynchronize() )

    stopwatchStop(&(s->execGpuRun));

    /** copy data back **/
    stopwatchStart(&(s->execMemcopyFromDevice));

    CE( cuMemcpyDtoH( s->info_space, deviceGlobalFreePointer, sizeof( *(s->info_space) ) ) )
    heap_end_long = s->info_space[0];
    heap_end_long <<= 4; // mul 16 ?
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
}

#undef CE
