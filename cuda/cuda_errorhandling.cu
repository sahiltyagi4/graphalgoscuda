#include "stdio.h"

__global__ void error_handle(int *a, int N){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;
    // gets error in the first for loop as we try to access element outside a's range
    //for (int j=0; j<N+stride; j+=stride){a[j] *= 2;}
    for (int j=0; j<N; j+=stride){a[j] *= 2;}
}

bool verify_doubling(int *a, int N){
    for (int k=0; k<N; k+=1){
        if (a[k] != 2*k){return false;}
    }
    return true;
}

int main(){
    int N = 1000000;
    size_t threads = 128;
    size_t blocks = (N + threads - 1)/threads;
    int *a;
    size_t size = N * sizeof(int);
    cudaMallocManaged(&a, size);
    cudaError_t sync_err, async_err;

    for (int i=0; i<N; i++){a[i] = i;}

    error_handle<<<blocks, threads>>>(a, N);

    sync_err = cudaGetLastError();
    async_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess){ printf('got error %s\n', cudaGetErrorString(sync_err));}
    if (async_err != cudaSuccess){ printf('got error %s\n', cudaGetErrorString(async_err));}

    bool is_true = verify_doubling(a, N);
    print(is_true);
    cudaFree(a);
}