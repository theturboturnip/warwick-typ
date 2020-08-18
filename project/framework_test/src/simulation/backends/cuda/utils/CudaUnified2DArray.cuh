//
// Created by samuel on 12/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vector>

#include <cuda_runtime.h>
#include "simulation/backends/cuda/kernels/common.cuh"

enum class CudaMemoryType {
    CudaManaged,
    Native
};

template<typename T, CudaMemoryType MemoryType=CudaMemoryType::CudaManaged>
// TODO - Type erasure for MemoryType? It could be feasible to expect all things to use the same MemoryType
class CudaUnified2DArray {
    //static_assert(!PitchedAlloc, "PitchedAlloc hasn't been implemented yet");
public:
    CudaUnified2DArray() : width(0), height(0) {}
    explicit CudaUnified2DArray(Size<size_t> size) : CudaUnified2DArray(size.x, size.y) {}
    explicit CudaUnified2DArray(size_t width, size_t height) : width(width), height(height) {
        // TODO - pitch allocation
        //  pitched arrays MUST have zeroes in the padding for reductions to work - note this
        if (MemoryType == CudaMemoryType::CudaManaged) {
            cudaMallocManaged(&raw_data, width * height * sizeof(T));
            col_pitch = height;
            raw_length = width * height;
        } else {
            raw_data = (T*)malloc(width * height * sizeof(T));
            col_pitch = height;
            raw_length = width * height;
        }

        cpu_pointers = std::vector<T*>();
        for (int i = 0; i < width; i++) {
            cpu_pointers.push_back(raw_data + (i * col_pitch));
        }
    }
    CudaUnified2DArray(const CudaUnified2DArray<T>&) = delete;
    ~CudaUnified2DArray() {
        if (raw_data) {
            if (MemoryType == CudaMemoryType::CudaManaged) {
                cudaFree(raw_data);
            } else {
                free(raw_data);
            }
        }
    }

    void dispatch_gpu_prefetch(int dstDevice, cudaStream_t stream) {
        cudaMemPrefetchAsync(raw_data, raw_length*sizeof(T), dstDevice, stream);
    }

    template<typename = typename std::enable_if<MemoryType == CudaMemoryType::CudaManaged>::type>
    T* as_gpu() {
        static_assert(MemoryType == CudaMemoryType::CudaManaged, "as_gpu() only exists when NativeMemOnly = false!");
        return raw_data;
    }
    T** as_cpu() {
        return cpu_pointers.data();
    }

    void zero_out() {
        memset(raw_data, 0, raw_length*sizeof(T));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == raw_length);
        // TODO - Use cudaMemcpy here?
        memcpy(raw_data, new_data.data(), raw_length * sizeof(T));
    }
    void memcpy_in(const CudaUnified2DArray<T>& other) {
        DASSERT(other.raw_length == raw_length);
        cudaMemcpy(raw_data, other.raw_data, raw_length*sizeof(T), cudaMemcpyDefault);
    }
    void dispatch_memcpy_in(const CudaUnified2DArray<T>& other, cudaStream_t stream) {
        DASSERT(other.raw_length == raw_length);
        cudaMemcpyAsync(raw_data, other.raw_data, raw_length*sizeof(T), cudaMemcpyDefault, stream);
    }
    std::vector<T> extract_data() {
        return std::vector<T>(raw_data, raw_data + raw_length);
    }

    const size_t width, height;
    size_t col_pitch;
    size_t raw_length;
private:
    T* raw_data = nullptr;

    std::vector<T*> cpu_pointers;
};