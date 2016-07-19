
#ifndef NAN
#   include <math_constants.h>
#   define NAN CUDART_NAN
#endif

#ifndef INFINITY
#   include <math_constants.h>
#   define INFINITY CUDART_INF
#endif

#include <stdio.h>
#include <assert.h>


__constant__ size_t m_Local[3];
__shared__ char m_shared[%%shared_mem_size%%];


__device__ int getThreadId()
{
    int linearized = 0;
    int max        = 1;

    // this ordering seems highly unorthodox to me.
    // Normally threadIdx.x would be the least significant number
    linearized += threadIdx.z * max; max *= blockDim.z;
    linearized += threadIdx.y * max; max *= blockDim.y;
    linearized += threadIdx.x * max; max *= blockDim.x;
    linearized += blockIdx.y  * max; max *= gridDim.y;
    linearized += blockIdx.x  * max;

    return linearized;
}

__device__ int getThreadIdxx(){ return threadIdx.x; }
__device__ int getThreadIdxy(){ return threadIdx.y; }
__device__ int getThreadIdxz(){ return threadIdx.z; }
__device__ int getBlockIdxx (){ return blockIdx.x ; }
__device__ int getBlockIdxy (){ return blockIdx.y ; }
__device__ int getBlockDimx (){ return blockDim.x ; }
__device__ int getBlockDimy (){ return blockDim.y ; }
__device__ int getBlockDimz (){ return blockDim.z ; }
__device__ long long getGridDimx(){ return gridDim.x; }
__device__ long long getGridDimy(){ return gridDim.y; }
__device__ void org_trifort_syncthreads(){ __syncthreads(); }
__device__ int  org_trifort_syncthreads_count(int value){ return __syncthreads_count(value); }
__device__ void org_trifort_threadfence(){ __threadfence(); }
__device__ void org_trifort_threadfence_block(){ __threadfence_block(); }
__device__ void org_trifort_threadfence_system(){ __threadfence_system(); }

__device__ clock_t global_now;
