#include <stdio.h>

__global__ void runGPU(int *a, int N) {
    //printf("array size %d\n", (int)sizeof(a));
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int  j=ix; j<N; j+=stride){a[j] = 2 * a[j];} 
}

int main(void) {
      int N = 1<<20;
      int *a;
      size_t size = N * sizeof(int);
      //cudaMallocManaged(&a, size);
      cudaMallocHost(&a, size);
      for (int i=0; i<N; i++){
        a[i] = i;
      }
      runGPU<<<1,128>>>(a, N);
      cudaDeviceSynchronize();
      cudaFree(a);
}
