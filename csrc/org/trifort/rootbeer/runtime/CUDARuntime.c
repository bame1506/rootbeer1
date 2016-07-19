
#include "CUDARuntime.h"
#include <cuda.h>
#include <assert.h>
#include <stdio.h>      // sprintf


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
        assert( status == CUDA_SUCCESS );                           \
    }                                                               \
}


/**
 * Basically the same functionality like cudaGetDeviceProperties looped over
 * cudaGetDeviceCount. Not sure why CUDA runtime is used directly. Faster?
 * Basically 95% boiler plate code
 *
 * Uses JNI to create an ArrayList[GpuDevice] object
 */
JNIEXPORT jobject JNICALL
Java_org_trifort_rootbeer_runtime_CUDARuntime_loadGpuDevices
(
    JNIEnv * env,
    jobject this_ref
)
{
    jclass    gpu_device_class;
    jmethodID gpu_device_init ;
    jobject   gpu_device      ;

    jclass    array_list_class = (*env)->FindClass  ( env, "java/util/ArrayList" );
    jmethodID array_list_init  = (*env)->GetMethodID( env, array_list_class, "<init>", "()V" );
    jmethodID array_list_add   = (*env)->GetMethodID( env, array_list_class, "add", "(Ljava/lang/Object;)Z" );
    jobject   ret              = (*env)->NewObject  ( env, array_list_class, array_list_init );

    gpu_device_class = (*env)->FindClass(env, "org/trifort/rootbeer/runtime/GpuDevice");
    gpu_device_init  = (*env)->GetStaticMethodID(env, gpu_device_class, "newCudaDevice",
      "(IIILjava/lang/String;JJIIIIIIIIZIIIIIIII)Lorg/trifort/rootbeer/runtime/GpuDevice;");
                          /* ^ function signature for constructor arguments */

    int status = cuInit(0);
    if ( status != CUDA_SUCCESS )
        return ret;

    int nDevices = 0;
    cuDeviceGetCount( &nDevices );

    int iDevice = 0;
    for ( iDevice = 0; iDevice < nDevices; ++iDevice )
    {
        CUdevice device;
        status = cuDeviceGet( &device, iDevice );
        if ( status != CUDA_SUCCESS )
            continue;

        int    major_version    ;
        int    minor_version    ;
        char   device_name[4096];
        size_t free_mem         ;
        size_t total_mem        ;
        CUcontext context       ;

        CE( cuDeviceComputeCapability( &major_version, &minor_version, device ) );
        CE( cuDeviceGetName( device_name, 4096, device ) );
        CE( cuCtxCreate ( &context, CU_CTX_MAP_HOST, device ) );
        CE( cuMemGetInfo( &free_mem, &total_mem ) );
        CE( cuCtxDestroy( context ) );

        /* makes use of https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
         * to be able to write the constructor call, variable declaration and
         * call to cuDeviceGetAttribute in one go using a macro.
         * Also seems to work with -std=c99 switch:
         *   f(
         *       ({ int j = 5; j+1; })
         *   );
         */
        #define CUATTR( NAME )                                      \
        ( {                                                         \
            int NAME;                                               \
            CE( cuDeviceGetAttribute( &NAME,                        \
                                      CU_DEVICE_ATTRIBUTE_##NAME,   \
                                      device ) )                    \
            NAME;                                                   \
        } )

        /* @see GpuDevice.java */
        gpu_device = (*env)->CallObjectMethod
        (
            env                                     ,
            gpu_device_class                        ,
            gpu_device_init                         ,
            iDevice                                 ,  // device_id
            major_version                           ,  // major_version
            minor_version                           ,  // minor_version
            (*env)->NewStringUTF(env, device_name)  ,  // device_name
            (jlong) free_mem                        ,  // free_global_mem_size
            (jlong) total_mem                       ,  // total_global_mem_size
            CUATTR( MAX_REGISTERS_PER_BLOCK        ),  // max_registers_per_block
            CUATTR( WARP_SIZE                      ),  // warp_size
            CUATTR( MAX_PITCH                      ),  // max_pitch
            CUATTR( MAX_THREADS_PER_BLOCK          ),  // max_threads_per_block
            CUATTR( MAX_SHARED_MEMORY_PER_BLOCK    ),  // max_shared_memory_per_block
            CUATTR( CLOCK_RATE                     ),  // clock_rate
            CUATTR( MEMORY_CLOCK_RATE              ),  // memory_clock_rate
            CUATTR( TOTAL_CONSTANT_MEMORY          ),  // constant_mem_size
            CUATTR( INTEGRATED                     ),  // integrated
            CUATTR( MAX_THREADS_PER_MULTIPROCESSOR ),  // max_threads_per_multiprocessor
            CUATTR( MULTIPROCESSOR_COUNT           ),  // multiprocessor_count
            CUATTR( MAX_BLOCK_DIM_X                ),  // max_block_dim_x
            CUATTR( MAX_BLOCK_DIM_Y                ),  // max_block_dim_y
            CUATTR( MAX_BLOCK_DIM_Z                ),  // max_block_dim_z
            CUATTR( MAX_GRID_DIM_X                 ),  // max_grid_dim_x
            CUATTR( MAX_GRID_DIM_Y                 ),  // max_grid_dim_y
            CUATTR( MAX_GRID_DIM_Z                 )   // max_grid_dim_z
        );

        #undef CUATTR

        (*env)->CallBooleanMethod( env, ret, array_list_add, gpu_device );
    }

    return ret;
}
