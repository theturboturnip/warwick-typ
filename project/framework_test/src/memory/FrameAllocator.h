//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "memory/internal/MType.h"
#include "memory/internal/FrameAllocator_fwd.h" // To ensure we match the fwd decl.
#include "memory/internal/Sim2DArray.h"
#include "memory/internal/SimRedBlackArray.h"

#include "util/Size.h"

#if CUDA_ENABLED
#include <cuda_runtime_api.h>
#include <rendering/vulkan/helpers/VulkanDeviceMemory.h>
#include <rendering/vulkan/VulkanContext.h>
#include <memory/vulkan/VulkanCudaBufferMemory.h>
#include "util/check_cuda_error.cuh"
#endif

/**
 * This file defines a set of FrameAllocators.
 * These allocate memory per-frame for simulations, from various sources.
 * Currently, CPU memory (malloc/free), CUDA managed memory, and Vulkan/Cuda shared memory are available.
 *
 * Each FrameAllocator has a common interface:
 * Sim2DArray<T, MType> allocate2D(Size<uint32_t>) allocates a Sim2DArray of the specified size.
 * SimRedBlackArray<T, MType, Storage> allocateRedBlack(Size<uint32_t>) allocates a SimRedBlackArray of the specified size.
 * FrameAllocators own the memory they allocate so they cannot be copied, only moved.
 */


/**
 * Template specialization of FrameAllocator for CPU memory. Simply uses malloc() and free().
 * Frees data on destruct.
 */
template<>
class FrameAllocator<MType::Cpu> {
public:
    template<class T>
    Sim2DArray<T, MType::Cpu> allocate2D(Size<uint32_t> otherSize) {
        return allocate2D_internal<T>(otherSize);
    }

    FrameAllocator() = default;
    FrameAllocator(FrameAllocator&&) = default;
    FrameAllocator(const FrameAllocator&) = delete;

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::Cpu, Storage> allocateRedBlack(Size<uint32_t> paddedSize) {
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D_internal<T>(splitSize);
        auto black = allocate2D_internal<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D_internal<T>(paddedSize);

            return SimRedBlackArray<T, MType::Cpu, RedBlackStorage::WithJoined>(
                    std::move(joined),
                    std::move(red),
                    std::move(black)
            );
        } else {
            return SimRedBlackArray<T, MType::Cpu, RedBlackStorage::RedBlackOnly>(
                    std::move(red),
                    std::move(black)
            );
        }
    }

    ~FrameAllocator() {
        for (void* ptr : allocatedPtrs) {
            free(ptr);
        }
        allocatedPtrs.clear();
    }

private:
    template<class T>
    Sim2DArray<T, MType::Cpu> allocate2D_internal(Size<uint32_t> size) {
        Sim2DArrayStats stats = {
                .width = size.x,
                .height = size.y,
                .col_pitch = size.y,
                .raw_length = size.x * size.y
        };

        T* data = static_cast<T*>(malloc(sizeof(T) * stats.raw_length));
        allocatedPtrs.push_back(data);

        return Sim2DArray<T, MType::Cpu>(stats, data);
    }

    std::vector<void*> allocatedPtrs;
};

#if CUDA_ENABLED
/**
 * Template specialization of FrameAllocator for CUDA Unified memory. Simply uses malloc() and free().
 * Frees data on destruct.
 */
template<>
class FrameAllocator<MType::Cuda> {
public:
    template<class T>
    Sim2DArray<T, MType::Cuda> allocate2D(Size<uint32_t> size) {
        return allocate2D_internal<T>(size);
    }

    FrameAllocator() = default;
    FrameAllocator(FrameAllocator&&) = default;
    FrameAllocator(const FrameAllocator&) = delete;

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::Cuda, Storage> allocateRedBlack(Size<uint32_t> paddedSize) {
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D_internal<T>(splitSize);
        auto black = allocate2D_internal<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D_internal<T>(paddedSize);

            return SimRedBlackArray<T, MType::Cuda, RedBlackStorage::WithJoined>(
                    std::move(joined),
                    std::move(red),
                    std::move(black)
            );
        } else {
            return SimRedBlackArray<T, MType::Cuda, RedBlackStorage::RedBlackOnly>(
                    std::move(red),
                    std::move(black)
            );
        }
        FATAL_ERROR("Can't get here");
    }

    ~FrameAllocator() {
        for (void* ptr : allocatedPtrs) {
            cudaFree(ptr);
        }
        allocatedPtrs.clear();
    }

private:
    template<class T>
    Sim2DArray<T, MType::Cuda> allocate2D_internal(Size<uint32_t> size) {
        Sim2DArrayStats stats = {
                .width = size.x,
                .height = size.y,
                .col_pitch = size.y,
                .raw_length = size.x * size.y
        };

        T* data = nullptr;
        CHECKED_CUDA(cudaMallocManaged(&data, sizeof(T) * stats.raw_length));
        allocatedPtrs.push_back(data);

        return Sim2DArray<T, MType::Cuda>(stats, data);
    }

    std::vector<void*> allocatedPtrs;
};


/**
 * Template specialization of FrameAllocator for Vulkan/Cuda shared memory.
 * Overall data for each frame is allocated all at once, then sub-allocated by allocate2D<T>().
 * See FrameSetAllocator for details on how the allocation bytes are calculated.
 * This means the restrictions are tighter for the frame - it must ONLY use VulkanCuda data for u,v,p,fluidmask.
 */
template<>
class FrameAllocator<MType::VulkanCuda> {
public:
    explicit FrameAllocator(VulkanContext& context, Size<uint32_t> paddedSize, size_t totalAllocationBytes)
            : paddedSize(paddedSize),
                    memory(context, totalAllocationBytes),
                    bytesUsed(0) {}

    FrameAllocator(FrameAllocator&&) = default;
    FrameAllocator(const FrameAllocator&) = delete;

    Size<uint32_t> paddedSize;

    template<class T>
    Sim2DArray<T, MType::VulkanCuda> allocate2D(Size<uint32_t> size) {
        FATAL_ERROR_UNLESS(size == paddedSize, "FrameAllocator<VulkanCuda> expects all 2D allocations to be of size %d %d\n", paddedSize.x, paddedSize.y);
        return allocate2D_internal<T>(paddedSize);
    }

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::VulkanCuda, Storage> allocateRedBlack(Size<uint32_t> size) {
        FATAL_ERROR_UNLESS(size == paddedSize, "FrameAllocator<VulkanCuda> expects all 2D allocations to be of size %d %d\n", paddedSize.x, paddedSize.y);
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D_internal<T>(splitSize);
        auto black = allocate2D_internal<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D_internal<T>(paddedSize);

            return SimRedBlackArray<T, MType::VulkanCuda, RedBlackStorage::WithJoined>(
                    std::move(joined),
                    std::move(red),
                    std::move(black)
            );
        } else {
            return SimRedBlackArray<T, MType::VulkanCuda, RedBlackStorage::RedBlackOnly>(
                    std::move(red),
                    std::move(black)
            );
        }
    }

private:
    template<class T>
    Sim2DArray<T, MType::VulkanCuda> allocate2D_internal(Size<uint32_t> size) {
        // Create T* cudaPointer, and vk::DescriptorBufferInfo
        Sim2DArrayStats stats = {
                .width = size.x,
                .height = size.y,
                .col_pitch = size.y,
                .raw_length = size.x * size.y
        };

        // Cast to char* to get correct pointer arithmetic, then cast back to T
        T* data = (T*)((char*)memory.as_cuda() + bytesUsed);

        vk::DescriptorBufferInfo vulkanBufferInfo;
        vulkanBufferInfo.buffer = memory.as_buffer();
        vulkanBufferInfo.offset = bytesUsed;
        vulkanBufferInfo.range = sizeof(T) * stats.raw_length;

        // Check if this allocation is actually valid
        bytesUsed += sizeof(T) * stats.raw_length;
        FATAL_ERROR_IF(bytesUsed > memory.sizeBytes, "FrameAllocator<Vulkan> out of memory");
        FATAL_ERROR_IF(
                (static_cast<std::uintptr_t>(data) % alignof(T)) == 0,
                "FrameAllocator<Vulkan> allocated misaligned data pointer for %s", typeid(T).name()
        );

        // Done - return the memory
        return Sim2DArray<T, MType::VulkanCuda>(stats, data, vulkanBufferInfo);
    }

    VulkanCudaBufferMemory memory;
    size_t bytesUsed;
};
#endif

#include "internal/SimArray_Constructors.inl"