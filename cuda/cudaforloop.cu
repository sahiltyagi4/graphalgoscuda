#include "stdio.h"
#include "sys/time.h"

//nvcc -arch=sm_70 -o cudaforloop cudaforloop.cu
void loopCPU(int N){
    for (int i=0; i < N; i++){
        printf("%d\n",i);
    }
}

__global__ void cudaGPU(int N){
    int data_ix = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n",data_ix);
    for (int i=0; i<N; i++){printf("id%d\n",i);}
}

int main(){
    int N=100;
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    //loopCPU(N);
    cudaGPU<<<1,128>>>(N);
    cudaDeviceSynchronize();
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld seconds\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}
