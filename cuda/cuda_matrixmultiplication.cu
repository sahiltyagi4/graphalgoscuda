#include "stdio.h"
#include "assert.h"

void cpu_matmul(int *a, int *b, int *c, int N){
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            v=0;
            for (int k=0; k<N; k++){
                v += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = v;
        }
    }
}

void initialize(int *a, int *b, int *c, int N){
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            a[row * N + col] = row;
            b[row * N + col] = col;
            c[row * N + col] = 0;
        }
    }
}

__global__ void cuda_matmul(int *a, int *b, int *c, int N){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int val = 0;

    if (row< N && col< N){
        for (int k=0; k<N; k++){
            val += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = val;
    }
}

void checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        printf("cuda error:%s\n", cudeGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// cpu execution
int main(){
    int N = 100000;
    size_t size = N * N * sizeof(int);
    int *a, *b, *c;
    a = malloc(size);
    b = malloc(size);
    c = malloc(size);

    initialize(a, b, c, N);
    cpu_matmul(a, b, c, N);

    free(a);
    free(b);
    free(c);
}

// cuda execution
//int main(){
//    int N = 1000000;
//    // blockDim.x and blockDim.y = 16
//    dim3 threads_perblock(16,16,1);
//    // gridDim.x and gridDim.y = 16
//    //dim3 number_blocks(16,16,1);
//    dim3 number_of_blocks((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);
//
//    size_t size = N * N * sizeof(int);
//    int *a, *b, *c;
//    checkCuda(cudaMallocManaged(&a, size));
//    checkCuda(cudaMallocManaged(&b, size));
//    checkCuda(cudaMallocManaged(&c, size));
//
//    initialize(a, b, c, N);
//    cuda_matmul<<<number_of_blocks, threads_perblock>>>(a, b, c, N);
//
//    checkCuda(cudaGetLastError());
//    checkCuda(cudaDeviceSynchronize());
//
//    checkCuda(cudaFree(a));
//    checkCuda(cudaFree(b));
//    checkCuda(cudaFree(c));
//}