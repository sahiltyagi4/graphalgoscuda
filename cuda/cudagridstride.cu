#include "stdio.h"

__global__ void gridstrides_cuda(int *a, int N){
    int data_ix = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_stride = gridDim.x * blockDim.x;
    for (int i=data_ix; i<N; i += grid_stride) {
        print('doing grid strides of %d\n', grid_stride);
    }
}

int main(){
    int N = 1000000;
    size_t num_threads_perblock = 128;
    size_t num_blocks = (N + num_threads_perblock -1)/num_threads_perblock;

    size_t size = N * sizeof(int);
    int *a;
    cudaMallocManaged(&a, size);
    for (int i=0; i< N; i++){
        a[i] = i;
    }

    gridstrides_cuda<<<num_blocks, num_threads_perblock>>>(a, N);
    cudaDeviceSynchronize();
    cudaFree(a);
}