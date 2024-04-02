#include "stdio.h"
#include "sys/time.h"

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

//gpu increment
int main() {
    size_t num_blocks = 128;
    size_t num_threads_perblock = 512;
    int N = num_blocks * num_threads_perblock;
    size_t size = N * sizeof(int);
    int *a;
    cudaMallocManaged(&a, size);
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    //on_cpu(N);
    on_gpu<<<num_blocks, num_threads_perblock>>>(a, N);
    cudaDeviceSynchronize();
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld seconds\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    cudaFree(a);
    ////bool did_increment = verify_op(a, N);
    ////printf("did array elements increment on GPU %s\n", did_increment);
}
