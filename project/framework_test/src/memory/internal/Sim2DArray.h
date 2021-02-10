//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "MType.h"
#include "FrameAllocator_fwd.h"

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.hpp>

#if CUDA_ENABLED
#include <cuda_runtime_api.h>
#include "util/check_cuda_error.cuh"
#endif

#include "util/Size.h"
#include "util/fatal_error.h"

struct Sim2DArrayStats {
    const uint32_t width, height;
    uint32_t col_pitch;
    size_t raw_length;
};

/**
 * Template for non-owning 2D column-major memory view, represented by at least one MType memory type.
 * Memory is owned by the Allocator<memType> that creates it.
 *
 * @tparam T Type of data to store e.g. float
 * @tparam memType Kind of memory this views e.g. MType::VulkanCuda
 */
template<class T, MType MemType>
class Sim2DArray;


template<class T>
class Sim2DArray<T, MType::Cpu> {
public:
    Sim2DArrayStats stats;
    static size_t sizeBytesOf(Size<uint32_t> size) {
        return size.area() * sizeof(T);
    }
    constexpr static MType MemType = MType::Cpu;

    explicit Sim2DArray(FrameAllocator<MType::Cpu>& alloc, Size<uint32_t> size);

    const T** as_cpu() const {
        // cpu_pointers.data() returns const T* const* - that's inconvenient, so cast it.
        // This is terrible. Blame C++.
        return const_cast<const T**>(cpu_pointers.data());
    }
    T** as_cpu() {
        return cpu_pointers.data();
    }

    // CPU memory is not accessible on CUDA GPUs.
    const T* as_cuda() const = delete;
    T* as_cuda() = delete;

    vk::DescriptorBufferInfo as_vulkan() = delete;

    void zero_out() {
        memset(raw_data, 0, stats.raw_length * sizeof(T));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == stats.raw_length);
        memcpy(raw_data, new_data.data(), stats.raw_length * sizeof(T));
    }
    std::vector<T> extract_data() const {
        return std::vector<T>(raw_data, raw_data + stats.raw_length);
    }
private:
    T* raw_data;
    std::vector<T*> cpu_pointers;

    Sim2DArray(Sim2DArrayStats stats, T* data)
        : stats(stats),
          raw_data(data),
          cpu_pointers(stats.width)
        {
        for (uint32_t i = 0; i < stats.width; i++) {
            cpu_pointers[i] = data + i * stats.col_pitch;
        }
    }

    friend class FrameAllocator<MType::Cpu>;
};

#if CUDA_ENABLED
template<class T>
class Sim2DArray<T, MType::Cuda> {
public:
    Sim2DArrayStats stats;
    static size_t sizeBytesOf(Size<uint32_t> size) {
        return size.area() * sizeof(T);
    }
    constexpr static MType MemType = MType::Cuda;

    explicit Sim2DArray(FrameAllocator<MType::Cuda>& alloc, Size<uint32_t> size);

    const T** as_cpu() const {
        // cpu_pointers.data() returns const T* const* - that's inconvenient, so cast it.
        // This is terrible. Blame C++.
        return const_cast<const T**>(cpu_pointers.data());
    }
    T** as_cpu() {
        return cpu_pointers.data();
    }

    const T* as_cuda() const {
        return raw_data;
    }
    T* as_cuda() {
        return raw_data;
    }

    // Cuda Unified memory is not available to Vulkan
    vk::DescriptorBufferInfo as_vulkan() = delete;

    void zero_out() {
        CHECKED_CUDA(cudaMemset(raw_data, 0, stats.raw_length*sizeof(T)));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == stats.raw_length);
        CHECKED_CUDA(cudaMemcpy(raw_data, new_data.data(), stats.raw_length * sizeof(T), cudaMemcpyDefault));
    }
    template<MType OtherMemType>
    void memcpy_in(const Sim2DArray<T, OtherMemType>& other) {
        // .raw_data will *always* be a CUDA Unified-accessible pointer.
        // For CPU, it could be a CPU pointer which is CUDA-usable.
        // For CUDA, it will be a CUDA unified pointer.
        // For VulkanCUDA, it will be a normal CUDA pointer (not unified), which we can handle.
        DASSERT(other.stats.raw_length == stats.raw_length);
        CHECKED_CUDA(cudaMemcpy(raw_data, other.raw_data, stats.raw_length*sizeof(T), cudaMemcpyDefault));
    }
    template<MType OtherMemType>
    void dispatch_memcpy_in(const Sim2DArray<T, OtherMemType>& other, cudaStream_t stream) {
        DASSERT(other.stats.raw_length == stats.raw_length);
        CHECKED_CUDA(cudaMemcpyAsync(raw_data, other.raw_data, stats.raw_length*sizeof(T), cudaMemcpyDefault, stream));
    }
    void dispatch_gpu_prefetch(int dstDevice, cudaStream_t stream) {
        CHECKED_CUDA(cudaMemPrefetchAsync(raw_data, stats.raw_length*sizeof(T), dstDevice, stream));
    }
    std::vector<T> extract_data() const {
        return std::vector<T>(raw_data, raw_data + stats.raw_length);
    }

private:
    T* raw_data;
    std::vector<T*> cpu_pointers;

    Sim2DArray(Sim2DArrayStats stats, T* data)
            : stats(stats),
              raw_data(data),
              cpu_pointers(stats.width)
    {
        for (uint32_t i = 0; i < stats.width; i++) {
            cpu_pointers[i] = data + i * stats.col_pitch;
        }
    }

    friend class FrameAllocator<MType::Cuda>;
    // Let VulkanCuda Sim2DArray access our raw_data pointer
    friend class Sim2DArray<T, MType::VulkanCuda>;
};

template<class T>
class Sim2DArray<T, MType::VulkanCuda> {
public:
    Sim2DArrayStats stats;
    static size_t sizeBytesOf(Size<uint32_t> size) {
        return size.area() * sizeof(T);
    }
    constexpr static MType MemType = MType::VulkanCuda;

    explicit Sim2DArray(FrameAllocator<MType::VulkanCuda>& alloc, Size<uint32_t> size);

    // VulkanCuda memory is not CPU-accessible without mapping etc.
    const T** as_cpu() const = delete;
    T** as_cpu() = delete;

    const T* as_cuda() const {
        return raw_data;
    }
    T* as_cuda() {
        return raw_data;
    }

    const vk::DescriptorBufferInfo as_vulkan() const {
        return vulkanBufferInfo;
    }

    void zero_out() {
        CHECKED_CUDA(cudaMemset(raw_data, 0, stats.raw_length*sizeof(T)));
    }
    void memcpy_in(const std::vector<T>& new_data) {
        DASSERT(new_data.size() == stats.raw_length);
        CHECKED_CUDA(cudaMemcpy(raw_data, new_data.data(), stats.raw_length * sizeof(T), cudaMemcpyDefault));
    }
    template<MType OtherMemType>
    void memcpy_in(const Sim2DArray<T, OtherMemType>& other) {
        // .raw_data will *always* be a CUDA Unified-accessible pointer.
        // For CPU, it could be a CPU pointer which is CUDA-usable.
        // For CUDA, it will be a CUDA unified pointer.
        // For VulkanCUDA, it will be a normal CUDA pointer (not unified), which we can handle.
        DASSERT(other.stats.raw_length == stats.raw_length);
        CHECKED_CUDA(cudaMemcpy(raw_data, other.raw_data, stats.raw_length*sizeof(T), cudaMemcpyDefault));
    }
    template<MType OtherMemType>
    void dispatch_memcpy_in(const Sim2DArray<T, OtherMemType>& other, cudaStream_t stream) {
        DASSERT(other.stats.raw_length == stats.raw_length);
        CHECKED_CUDA(cudaMemcpyAsync(raw_data, other.raw_data, stats.raw_length*sizeof(T), cudaMemcpyDefault, stream));
    }
    void dispatch_gpu_prefetch(int dstDevice, cudaStream_t stream) = delete; // Only works on Unified Memory

    std::vector<T> extract_data() const {
        auto vec = std::vector<T>(stats.raw_length);
        CHECKED_CUDA(cudaMemcpy(vec.data(), raw_data, stats.raw_length * sizeof(T), cudaMemcpyDeviceToHost));
        return vec;
    }

private:
    T* raw_data;
    vk::DescriptorBufferInfo vulkanBufferInfo;

    Sim2DArray(Sim2DArrayStats stats, T* data, vk::DescriptorBufferInfo vulkanBufferInfo)
            : stats(stats),
              raw_data(data),
              vulkanBufferInfo(vulkanBufferInfo)
    {}

    friend class FrameAllocator<MType::VulkanCuda>;
    // Let Cuda Sim2DArray access our raw_data pointer
    friend class Sim2DArray<T, MType::Cuda>;
};
#endif