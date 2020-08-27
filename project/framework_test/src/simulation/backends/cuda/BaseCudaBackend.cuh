//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <util/check_cuda_error.cuh>

#include "simulation/memory/CudaUnified2DAllocator.cuh"

class BaseCudaBackend {
protected:
    BaseCudaBackend() : unifiedAlloc(std::make_unique<CudaUnified2DAllocator>()) {
        CHECKED_CUDA(cudaStreamCreate(&stream));
    }
    ~BaseCudaBackend() {
        CHECKED_CUDA(cudaStreamDestroy(stream));
    }

    std::unique_ptr<CudaUnified2DAllocator> unifiedAlloc;
    cudaStream_t stream;
};