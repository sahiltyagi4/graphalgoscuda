#include "stdio.h"

void loopCPU(int N){
    for (int i=0; i < N; i++){
        printf('%d\n',i);
    }
}

__global__ void cuda_GPU(){
    int data_ix = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n",data_ix);
}

int main(){
    int N=1000000;
    loopCPU(N);
    cuda_GPU<<<1,N>>>();
    cudaDeviceSynchronize();
}