#include "stdio.h"
#include "assert.h"
#include "sys/time.h"

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

/*
inline cudaError_t checkCudaErrors(cudaError_t result){
    if (result != cudaSuccess){
        //printf("cuda error:%s\n", cudeGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}
*/

__global__ void cuda_vector_addition(int *a, int *b, int *c, int N){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int  j=ix; j<N; j+=stride){c[j] = a[j] + b[j];}
}

int main(){
    int N = 10000000;
    int *a, *b, *c;
    size_t threads = 128;
    size_t blocks = (N + threads -1)/threads;
    size_t size = N * sizeof(int);
    struct timeval tval_before, tval_after, tval_result;

    ////checkCudaErrors(cudaMallocManaged(&a, size));
    ////checkCudaErrors(cudaMallocManaged(&b, size));
    ////checkCudaErrors(cudaMallocManaged(&c, size));
    //cudaMallocManaged(&a, size);
    //cudaMallocManaged(&b, size);
    //cudaMallocManaged(&c, size);
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    initialize(a,b,c,N);
    
    gettimeofday(&tval_before, NULL);
    ////cuda_vector_addition<<<blocks, threads>>>(a, b, c, N);
    //cuda_vector_addition<<<1, N>>>(a, b, c, N);
    //cudaDeviceSynchronize();
    //cudaFree(a);
    //cudaFree(b);
    //cudaFree(c)

    ////checkCudaErrors(cudaGetLastError());
    ////checkCudaErrors(cudaDeviceSynchronize());

    ////checkCudaErrors(cudaFree(a));
    ////checkCudaErrors(cudaFree(b));
    ////checkCudaErrors(cudaFree(c));

    cpu_vector_addition(a, b, c, N);
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld seconds\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}
