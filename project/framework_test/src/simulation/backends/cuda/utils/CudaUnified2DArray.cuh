//
// Created by samuel on 12/08/2020.
//

#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "simulation/memory/I2DAllocator.h"
#include "util/fatal_error.h"
#include "util/check_cuda_error.cuh"

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

    T* as_gpu() {
        return raw_data;
    }
    T** as_cpu() {
        static_assert(UnifiedMemory, "Cannot get CPU pointers for not-unified data!");
        return cpu_pointers.data();
    }

    void zero_out() {
        CHECKED_CUDA(cudaMemset(raw_data, 0, raw_length*sizeof(T)));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == raw_length);
        CHECKED_CUDA(cudaMemcpy(raw_data, new_data.data(), raw_length * sizeof(T), cudaMemcpyDefault));
    }
    template<bool OtherUnifiedMemory>
    void memcpy_in(const CudaUnified2DArray<T, OtherUnifiedMemory>& other) {
        DASSERT(other.raw_length == raw_length);
        CHECKED_CUDA(cudaMemcpy(raw_data, other.raw_data, raw_length*sizeof(T), cudaMemcpyDefault));
    }
    template<bool OtherUnifiedMemory>
    void dispatch_memcpy_in(const CudaUnified2DArray<T, OtherUnifiedMemory>& other, cudaStream_t stream) {
        DASSERT(other.raw_length == raw_length);
        CHECKED_CUDA(cudaMemcpyAsync(raw_data, other.raw_data, raw_length*sizeof(T), cudaMemcpyDefault, stream));
    }
    void dispatch_gpu_prefetch(int dstDevice, cudaStream_t stream) {
        static_assert(UnifiedMemory, "cudaMemPrefetchAsync only works on Unified Memory");
        CHECKED_CUDA(cudaMemPrefetchAsync(raw_data, raw_length*sizeof(T), dstDevice, stream));
    }
    std::vector<T> extract_data() {
        if constexpr (UnifiedMemory) {
            return std::vector<T>(raw_data, raw_data + raw_length);
        } else {
            auto vec = std::vector<T>(raw_length);
            CHECKED_CUDA(cudaMemcpy(vec.data(), raw_data, raw_length * sizeof(T), cudaMemcpyDeviceToHost));
            return vec;
        }
        FATAL_ERROR("We shouldn't get to this point\n");
    }

    // TODO - These are all copies of elements of AllocatedMemory<T>. Remove them
    const uint32_t width, height;
    uint32_t col_pitch;
    size_t raw_length;
private:
    T* raw_data = nullptr;
    AllocatedMemory<T> memory;

    std::vector<T*> cpu_pointers;

    friend class CudaUnified2DArray<T, !UnifiedMemory>;
};