//
// Created by samuel on 18/06/2020.
//

#include "cuda_memory_wrappers.h"

#include <cstdint>

#include "cuda_vulkan_memory_wrappers.h"

// Define the constructors/destructors for CUDAUnified1DArray here, in a .cu file.
// This means any files that end up including the header, such as files connected to Simulation.h, aren't required to be CUDA-aware.
// TODO: This is dumb and unnecessary

template<typename T>
CUDAUnified1DArray<T>::CUDAUnified1DArray(size_t elemCount) : elemCount(elemCount) {
    cudaMallocManaged(&pointer, elemCount * sizeof(T));
}

template<typename T>
CUDAUnified1DArray<T>::~CUDAUnified1DArray<T>() {
    cudaFree(pointer);
}

// Explicitly instantiate the CUDAUnified1DArray with valid arguments.
// This ensures the relevant functions exist when other parts of the code try to use these classes,
// even if those parts of code were compiled from .cpp files.
template class CUDAUnified1DArray<float>;
template class CUDAUnified1DArray<double>;
template class CUDAUnified1DArray<uint32_t>;