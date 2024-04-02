#include "stdio.h"
#include "sys/time.h"

__global__ void gridstrides_cuda(int *a, int N){
    int data_ix = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_stride = gridDim.x * blockDim.x;
    int j=0;
    for (int i=data_ix; i<N; i += grid_stride) {
        //printf("doing grid strides of %d\n", grid_stride);
        j++;
    }
    printf("ThreadID %d made %d iterations\n", threadIdx.x, j);
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

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    //gridstrides_cuda<<<num_blocks, num_threads_perblock>>>(a, N);
    gridstrides_cuda<<<128, 1024>>>(a, N);
    cudaDeviceSynchronize();
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld seconds\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    cudaFree(a);
}
