#include <stdio.h>

//nvcc -arch=sm_70 -o test-gpu testgpu.cu -run
void runCPU() {
    printf("executing on CPU.\n")
}

__global__ void runGPU() {
    if(threadIdx.x == 127 && blockIdx.x == 3) {
        printf('executing on GPU.\n')
    }
}

int main() {
    runCPU();
    runGPU<<<4,128>>>();
    cudaDeviceSynchronize();
}