//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <cuda_runtime_api.h>
#include <util/check_cuda_error.cuh>

class BaseCudaBackend {
protected:
    BaseCudaBackend() {
        CHECKED_CUDA(cudaStreamCreate(&stream));
    }
    ~BaseCudaBackend() {
        CHECKED_CUDA(cudaStreamDestroy(stream));
    }

public: // TODO - make this use a more controlled access model?
    cudaStream_t stream;
};