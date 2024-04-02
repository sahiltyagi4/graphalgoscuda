#include "stdio.h"

__global__ void cudaEventLoop(int N){
    for (int i=0; i<N; i++) {
        printf("index %d\n", i);
    }
}

int main(){
    int N = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventLoop<<<4,128>>>(N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f milliseconds\n", milliseconds);
}
