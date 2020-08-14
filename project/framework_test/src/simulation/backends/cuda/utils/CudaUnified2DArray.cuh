//
// Created by samuel on 12/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vector>

#include <cuda_runtime.h>
#include "simulation/backends/cuda/kernels/common.cuh"

template<typename T, bool NativeMemOnly=false>
class CudaUnified2DArray {
public:
    CudaUnified2DArray() : width(0), height(0) {}
    explicit CudaUnified2DArray(Size<size_t> size) : width(size.x), height(size.y) {
        // TODO - pitch allocation
        if (NativeMemOnly) {
            raw_data = (T*)malloc(width * height * sizeof(T));
        } else {
            cudaMallocManaged(&raw_data, width * height * sizeof(T));
        }

        cpu_pointers = std::vector<T*>();
        for (int i = 0; i < width; i++) {
            cpu_pointers.push_back(raw_data + (i * height));
        }
    }
    CudaUnified2DArray(const CudaUnified2DArray<T>&) = delete;
    ~CudaUnified2DArray() {
        if (raw_data) {
            if (NativeMemOnly) {
                free(raw_data);
            } else {
                cudaFree(raw_data);
            }
        }
    }

    template<typename = typename std::enable_if<!NativeMemOnly>::type>
    T* as_gpu() {
        static_assert(!NativeMemOnly, "as_gpu() only exists when NativeMemOnly = false!");
        return raw_data;
    }
    T** as_cpu() {
        return cpu_pointers.data();
    }

    void zero_out() {
        memset(raw_data, 0, width*height*sizeof(T));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == width * height);
        memcpy(raw_data, new_data.data(), width * height * sizeof(T));
    }
    std::vector<T> extract_data() {
        return std::vector<T>(raw_data, raw_data + (width * height));
    }

    const size_t width, height;
    size_t col_pitch;
private:
    T* raw_data = nullptr;

    size_t raw_data_pitch = 0;
    std::vector<T*> cpu_pointers;
};