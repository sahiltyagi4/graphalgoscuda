#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <limits>
#include <string>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;
using std::cout;
using std::endl;
using namespace std::chrono;

void load_data(const char *f, std::vector<int> &vec){
    std::ifstream inputfile;
    int n;
    inputfile.open(f);
    while(inputfile.ignore() && (inputfile >> n)){
        vec.push_back(n);
    }
    cout<< "vector size:" << vec.size() << "vector file:" << f << endl;
    inputfile.close();
}

__global__ void initialize(int N, int *p, int val, bool src, int source, int sourceVal){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i=ix; i<N; i+= stride){
        p[i] = val;
        if(src){
            if(index == source) {
                p[i] = sourceVal;
            }
        }
    }
}

__global__ void resetGridStride(int N, int MAX_VAL, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i=ix; i< N-1; i+=stride){
        for (int j = d_in_I[i]; j < d_in_I[i + 1]; j++) {
            int w = d_in_W[j];
            int du = d_out_D[i];
            int dv = d_out_D[d_in_E[j]];
            int newDist = du + w;
            if (du == MAX_VAL){
                newDist = MAX_VAL;
            }

            if (newDist < dv) {
                atomicMin(&d_out_Di[d_in_E[j]],newDist);
            }
        }
    }
}

__global__ void update_distanceGridStride(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i=ix; i< N-1; i+=stride){
        if (d_out_D[i] > d_out_Di[i]) {
            d_out_D[i] = d_out_Di[i];
        }
        d_out_Di[i] = d_out_D[i];
    }
}

__global__ void update_INDEX_edgesGridStride(int N, int *d_in_V, int *d_in_E, int l, int r){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    int left = l;
    int right = r;
    for (int i=ix; i< N-1; i+=stride){
        left = l;
        right = r;
        while (left <= right) {
            int m = left + (right - left) / 2;
            if (d_in_V[m] == d_in_E[i]) {
                d_in_E[index] = m;
                break;
            }
            if (d_in_V[m] < d_in_E[i]) {
                left = m + 1;
            } else {
                right = m - 1;
            }
        }
    }
}

__global__ void update_resultsGridStride(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_P){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i=ix; i< N-1; i+=stride){
        for (int j = d_in_I[i]; j < d_in_I[i+1]; ++j) {
            int u = d_in_V[i];
            int w = d_in_W[j];
            int dis_u = d_out_D[i];
            int dis_v = d_out_D[d_in_E[j]];
            if (dis_v == dis_u + w) {
                atomicMin(&d_out_P[d_in_E[j]], u);
            }
        }
    }
}

int main(){
    std::vector<int> V, I, E, W;
    load_data("/home/styagi/rand_1000.gr_W.csv", V);
    load_data("/home/styagi/rand_1000.gr_W.csv", I);
    load_data("/home/styagi/rand_1000.gr_W.csv", E);
    load_data("/home/styagi/rand_1000.gr_W.csv", W);

//    load_data("/home/styagi/USA-road-d.NY.gr_W.csv", V);
//    load_data("/home/styagi/USA-road-d.NY.gr_W.csv", I);
//    load_data("/home/styagi/USA-road-d.NY.gr_W.csv", E);
//    load_data("/home/styagi/USA-road-d.NY.gr_W.csv", W);

    auto start_time = high_resolution_clock::now();
    cout<< "loaded vectors from file now copying to gpu and executing sssp" << endl;
    cout<< V.size() <<endl;

    int threads_per_block = 128;
    int max_value = std::numeric_limits<int>::max();
    int N = I.size();
    int blocks = (N + threads_per_block - 1)/threads_per_block;
    // execute on GPU:0
    cudaSetDevice(0);
    cout << "number of blocks: " << blocks << " Threads per block: " << threads_per_block << endl;

    int *d_in_V, *d_in_I, *d_in_E, *d_in_W, *d_out_D, *d_out_Di, *d_out_P;
    cudaMalloc((void**) &d_in_V, V.size() *sizeof(int));
    cudaMalloc((void**) &d_in_I, I.size() *sizeof(int));
    cudaMalloc((void**) &d_in_E, E.size() *sizeof(int));
    cudaMalloc((void**) &d_in_W, W.size() *sizeof(int));
    cudaMalloc((void**) &d_out_D, V.size() *sizeof(int));
    cudaMalloc((void**) &d_out_Di, V.size() *sizeof(int));
    cudaMalloc((void**) &d_out_P, V.size() *sizeof(int));

    cudaMemcpy(d_in_V, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_I, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_E, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_W, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    int init_blocks = (V.size() + threads_per_block -1) / threads_per_block;
    initialize<<<init_blocks, threads_per_block>>>(V.size(), d_out_D, max_value, true, 0, 0);
    initialize<<<init_blocks, threads_per_block>>>(V.size(), d_out_Di, max_value, true, 0, 0);
    initialize<<<init_blocks, threads_per_block>>>(V.size(), d_out_P, max_value, true, 0, 0);

    update_INDEX_edgesGridStride<<<init_blocks, threads_per_block>>>(E.size(), d_in_V, d_in_E, 0, V.size()-1);
    for (int i=0; i<V.size(); ++i){
        resetGridStride()<<<init_blocks, threads_per_block>>>(N, max_value, d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di);
        update_distanceGridStride<<<blocks, threads_per_block>>>(V.size(), d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di);
    }
    update_resultsGridStride<<<blocks, threads_per_block>>>(V.size(), d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_P);

    cout << "finished executing sssp on cuda..." << endl;
    cudaDeviceSynchronize();
    auto end_time = high_resolution_clock::now();
    auto exec_time = duration_cast<microseconds>(end_time - start_time);
    cout << exec_time.count() << "microseconds to complete" << endl;

    cudaFree(d_in_V);
    cudaFree(d_in_I);
    cudaFree(d_in_E);
    cudaFree(d_in_W);
    cudaFree(d_out_D);
    cudaFree(d_out_Di);
    cudaFree(d_out_P);

}