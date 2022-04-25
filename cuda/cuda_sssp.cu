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
    while(inputfile.ignore() && (inputfile >> n)){vec.push_back(n);}
    inputfile.close();
}

__global__ void reset(int N, int MAX_VAL, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < N-1){
        for (int j = d_in_I[ix]; j < d_in_I[ix + 1]; j++) {
            int w = d_in_W[j];
            int du = d_out_D[ix];
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

__global__ void update_distance(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < N) {
        if (d_out_D[ix] > d_out_Di[ix]) {d_out_D[ix] = d_out_Di[ix];}
        d_out_Di[ix] = d_out_D[ix];
    }
}

__global__ void update_INDEX_edges(int N, int *d_in_V, int *d_in_E, int l, int r){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    if (ix < N) {
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (d_in_V[m] == d_in_E[ix]) {
                d_in_E[index] = m;
                break;
            }
            if (d_in_V[m] < d_in_E[ix]) {l = m + 1;} else {r = m - 1;}
        }
    }
}

__global__ void initialize(int N, int *p, int val, bool src, int source, int sourceVal){
    int ix = threadIdx.x + (blockDim.x * blockIdx.x);
    if (ix < N) {
        p[ix] = val;
        if(src){
            if(ix == source) {
                p[ix] = sourceVal;
            }
        }
    }
}

__global__ void update_results(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_P){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    if (ix < N) {
        for (int j = d_in_I[ix]; j < d_in_I[ix+1]; ++j) {
            int u = d_in_V[ix];
            int w = d_in_W[j];
            int dis_u = d_out_D[ix];
            int dis_v = d_out_D[d_in_E[j]];
            if (dis_v == dis_u + w) {atomicMin(&d_out_P[d_in_E[j]], u);}
        }
    }
}

int main(){
    std::vector<int> V, I, E, W;
    load_data("/home/styagi/rand_1000.gr_V.csv", V);
    load_data("/home/styagi/rand_1000.gr_I.csv", I);
    load_data("/home/styagi/rand_1000.gr_E.csv", E);
    load_data("/home/styagi/rand_1000.gr_W.csv", W);

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    auto start_time = high_resolution_clock::now();
    cout<< "loaded vectors from file now copying to gpu and executing sssp" << endl;
    //cudaEventRecord(start, 0);

    int threads_per_block = 128;
    int max_value = std::numeric_limits<int>::max();
    int N = I.size();
    int blocks = (N + threads_per_block - 1)/threads_per_block;
    // execute on GPU:0
    cudaSetDevice(0);
    cout << "number of blocks: " << blocks << " Threads per block: " << threads_per_block << endl;

    int *d_in_V, *d_in_I, *d_in_E, *d_in_W, *d_out_D, *d_out_Di, *d_out_P;

//    cudaMallocManaged(d_in_V, V.size() * sizeof(int));
//    cudaMallocManaged(d_in_I, I.size() * sizeof(int));
//    cudaMallocManaged(d_in_E, E.size() * sizeof(int));
//    cudaMallocManaged(d_in_W, W.size() * sizeof(int));
//
//    cudaMallocManaged(d_out_D, V.size() * sizeof(int));
//    cudaMallocManaged(d_out_Di, V.size() * sizeof(int));
//    cudaMallocManaged(d_out_P, V.size() * sizeof(int));

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

    int e_blocks = (E.size() + threads_per_block -1) / threads_per_block;
    update_INDEX_edges<<<e_blocks, threads_per_block>>>(E.size(), d_in_V, d_in_E, 0, V.size()-1);

    for (int i=0; i<V.size(); ++i){
        reset<<<blocks, threads_per_block>>>(N, MAX_VAL, d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di);
        update_distance<<<blocks, threads_per_block>>>(V.size(), d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di);
    }
    update_results<<<blocks, threads_per_block>>>(V.size(), d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_P);

    //cudaEventRecord(stop, 0);
    //cudaEventSynchronization(stop);
    cout << "finished executing sssp on cuda..." << endl;
    cudaDeviceSynchronize();
    //float compute_time;
    //cudaEventElapsedTime(&compute_time, start, stop);
    auto end_time = high_resolution_clock::now();
    auto exec_time = duration_cast<microseconds>(end_time - start_time);
    cout << exec_time.count() << endl;

    cudaFree(d_in_V);
    cudaFree(d_in_I);
    cudaFree(d_in_E);
    cudaFree(d_in_W);
    cudaFree(d_out_D);
    cudaFree(d_out_Di);
    cudaFree(d_out_P);
}