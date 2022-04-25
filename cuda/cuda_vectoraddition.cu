#include "stdio.h"
#include "assert.h"

void cpu_vector_addition(int *a, int *b, int *c, int N){
    for (int i=0; i<N; i++){c[i] = a[i] + b[i];}
}

void initialize(int *a, int *b, int *c, int N){
    for (int i=0; i<N; i++){
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }
}

inline cudaError_t checkCudaErrors(cudaError_t result){
    if (result != cudaSuccess){
        printf("cuda error:%s\n", cudeGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void cuda_vector_addition(int *a, int *b, int *c, int N){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int  j=0; j<N; j+=stride){c[i] = a[i] + b[i];}
}

int main(){
    int N = 1000000;
    int *a, *b, *c;
    size_t threads = 128;
    size_t blocks = (N + threads -1)/threads;
    size_t size = N * sizeof(int);

    checkCudaErrors(cudaMallocManaged(&a, size));
    checkCudaErrors(cudaMallocManaged(&b, size));
    checkCudaErrors(cudaMallocManaged(&c, size));

    initialize(a,b,c,N);
    cuda_vector_addition<<<blocks, threads>>>(a, b, c, N);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));
    checkCudaErrors(cudaFree(c));
}