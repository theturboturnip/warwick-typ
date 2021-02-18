//
// Created by samuel on 18/02/2021.
//

#pragma once

#include <memory>

template<class T>
struct CudaUnifiedDeleter {
    void operator ()(T* ptr) {
        cudaFree(ptr);
    }
};

template<class T>
using CudaUniquePtr = std::unique_ptr<T, CudaUnifiedDeleter<T>>;

template<class T, class... Args>
CudaUniquePtr<T>&& make_cuda_unique(Args&&... args) {
    T* ptr = nullptr;
    cudaMalloc(&ptr, sizeof(T));
    DASSERT(ptr != nullptr);

    new(ptr) T(std::forward(args)...);

    return CudaUniquePtr<T>(ptr);
}