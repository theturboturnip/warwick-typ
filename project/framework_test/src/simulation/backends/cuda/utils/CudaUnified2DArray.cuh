//
// Created by samuel on 12/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vector>

#include <cuda_runtime.h>

template<typename T>
class CudaUnified2DArray {
public:
    CudaUnified2DArray() : width(0), height(0) {}// = default;
    explicit CudaUnified2DArray(Size<size_t> size) : width(size.x), height(size.y) {
        // TODO - pitch allocation
        cudaMallocManaged(&raw_data, width * height * sizeof(T));
        //raw_data = (T*)malloc(width * height * sizeof(T));
        cpu_pointers = std::vector<T*>();
        for (int i = 0; i < width; i++) {
            cpu_pointers.push_back(raw_data + (i * height));
        }
    }
    CudaUnified2DArray(const CudaUnified2DArray<T>&) = delete;
    ~CudaUnified2DArray() {
        if (raw_data) {
            cudaFree(raw_data);
            //free(raw_data);
        }
    }

    T* as_gpu() {
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

private:
    T* raw_data = nullptr;

    size_t raw_data_pitch = 0;
    std::vector<T*> cpu_pointers;
};