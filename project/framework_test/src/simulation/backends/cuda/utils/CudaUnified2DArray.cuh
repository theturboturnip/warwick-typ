//
// Created by samuel on 12/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vector>

//#include "simulation/backends/cuda/kernels/common.cuh"
#include <cuda_runtime.h>
#include <simulation/memory/I2DAllocator.h>


template<typename T, bool UnifiedMemory>
class CudaUnified2DArray {
public:
    CudaUnified2DArray(I2DAllocator* alloc, Size<uint32_t> size)
        : CudaUnified2DArray(alloc, alloc->allocate2D<T>(size))
    {}
    CudaUnified2DArray(I2DAllocator* alloc, AllocatedMemory<T> prealloc) : width(prealloc.width), height(prealloc.height){
        alloc->requireDeviceUsable();
        if constexpr (UnifiedMemory) {
            alloc->requireHostUsable();
        }

        memory = prealloc;
        raw_data = reinterpret_cast<T*>(memory.pointer);
        col_pitch = memory.columnStride;
        raw_length = memory.totalSize;

        // TODO - Is it worth specializing this class in order to not store CPU pointers on nonunified memory?
        cpu_pointers = std::vector<T*>();
        for (uint32_t i = 0; i < width; i++) {
            cpu_pointers.push_back(raw_data + (i * col_pitch));
        }
    }

    CudaUnified2DArray(const CudaUnified2DArray<T, UnifiedMemory>&) = delete;

    void dispatch_gpu_prefetch(int dstDevice, cudaStream_t stream) {
        cudaMemPrefetchAsync(raw_data, raw_length*sizeof(T), dstDevice, stream);
    }

    T* as_gpu() {
        return raw_data;
    }
    T** as_cpu() {
        static_assert(UnifiedMemory, "Cannot get CPU pointers for not-unified data!");
        return cpu_pointers.data();
    }

    void zero_out() {
        cudaMemset(raw_data, 0, raw_length*sizeof(T));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == raw_length);
        cudaMemcpy(raw_data, new_data.data(), raw_length * sizeof(T), cudaMemcpyDefault);
    }
    void memcpy_in(const CudaUnified2DArray<T, UnifiedMemory>& other) {
        DASSERT(other.raw_length == raw_length);
        cudaMemcpy(raw_data, other.raw_data, raw_length*sizeof(T), cudaMemcpyDefault);
    }
    void dispatch_memcpy_in(const CudaUnified2DArray<T, UnifiedMemory>& other, cudaStream_t stream) {
        DASSERT(other.raw_length == raw_length);
        cudaMemcpyAsync(raw_data, other.raw_data, raw_length*sizeof(T), cudaMemcpyDefault, stream);
    }
    std::vector<T> extract_data() {
        if constexpr (UnifiedMemory) {
            return std::vector<T>(raw_data, raw_data + raw_length);
        } else {
            auto vec = std::vector<T>(raw_length);
            cudaMemcpy(vec.data(), raw_data, raw_length * sizeof(T), cudaMemcpyDeviceToHost);
            return vec;
        }
    }

    // TODO - These are all copies of elements of AllocatedMemory<T>. Remove them
    const uint32_t width, height;
    uint32_t col_pitch;
    size_t raw_length;
private:
    T* raw_data = nullptr;
    AllocatedMemory<T> memory;

    std::vector<T*> cpu_pointers;
};