cmake_minimum_required(VERSION 3.19)
project(cuda_graphs CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_library(cuda_graphs library.cu library.cuh testgpu.cu cudaforloop.cu cudamemory.cu cudagridstride.cu cuda_errorhandling.cu cuda_vectoraddition.cu cuda_matrixmultiplication.cu cpu_sssp.cpp cuda_sssp.cu)

set_target_properties(
        cuda_graphs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)