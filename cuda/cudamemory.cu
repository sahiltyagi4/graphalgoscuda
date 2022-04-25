#include "stdio.h"

bool verify_op(int *a, int N){
    for (int i=0; i<N; i++){
        if (a[i] == i + 1){
            return true;
        } else{
            return false;
        }
    }
}

void on_cpu(int N){
    size_t size = N * sizeof(int);
    int *a;
    a = (int *)malloc(size);
    for (int i=0; i<N; i++){
        a[i] = i + 1;
    }
    bool did_increment = verify_op(a, N);
    printf("did increment happen on CPU? %s\n", did_increment);
    free(a);
}

__global__ void on_gpu(int *a, int N){
    printf("allocated CUDA unified memory..\n");
    int data_ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (data_ix < N){
        a[data_ix] = 1 + data_ix;
    }
}

// for testing increment on cpu
//int main(){
//    int N = 128 * 512;
//    on_cpu(N);
//}

//gpu increment
int main() {
    size_t num_blocks = 128;
    size_t num_threads_perblock = 512;
    int N = num_blocks * num_threads_perblock;
    size_t size = N * sizeof(int);
    int *a;
    cudaMallocManaged(&a, size);
    on_gpu<<<num_blocks, num_threads_perblock>>>(a, N);
    cudaDeviceSynchronize();
    bool did_increment = verify_op(a, N);
    printf("were array elements increment on GPU %s\n", did_increment);
    cudaFree();
}