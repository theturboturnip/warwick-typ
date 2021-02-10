//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "MType.h"
#include "FrameAllocator_fwd.h" // To ensure we match the fwd decl.
#include "Sim2DArray.h"
#include "SimRedBlackArray.h"

#include "util/Size.h"

#if CUDA_ENABLED
#include <cuda_runtime_api.h>
#include <rendering/vulkan/helpers/VulkanDeviceMemory.h>
#include <rendering/vulkan/VulkanContext.h>
#include <memory/vulkan/VulkanCudaBufferMemory.h>
#include "util/check_cuda_error.cuh"
#endif

/*
 // TODO - this has a lot of boilerplate right now.
 //    could be removed

 class FrameAllocator {
 private:
    unique_ptr<InternalAllocCpu> cpu;
    unique_ptr<InternalAllocCuda> cuda;
    unique_ptr<InternalAllocVulkan> vulkan;
 public:
    FrameAllocator(unique_ptr<Cpu> unique_ptr<Cuda> unique_ptr<Vulkan>)

    template<T, MemType>
    Sim2DArray<T, MemType> get(Size) {
        switch(MemType)
            case Cpu:
                return cpu->get<T>(size);
            etc.
    }

    template<T, MemType, joined?>
    SimRedBlackArray<T, MemType> get(Size) {
        Size halfSize = ...;
        red = get<T, MemType>(halfSize);
        black = --- as above ---;

        if (joined?) {
            joined = get<T, MemType>(fullSize);
            reutrn RedBlackArray(...);
        } else {
            return RedBlackArray;
        }
    }


class BasicFrameSetAllocator<TFrame> {
    BasicFrameSetAllocator() {
        foreach frame {
            frameAlloc = FrameAllocator(make_unique<Cpu>, make_unique<Cuda>);
            alloc frame...
        }
    }
}

class VulkanFrameSetAllocator<TFrame> {

 }

 PROBLEM: Vulkan not an easy compile-time switch
 */
// TODO remove unsized targets

template<>
class FrameAllocator<MType::Cpu> {
public:
    template<class T>
    Sim2DArray<T, MType::Cpu> allocate2D(Size<uint32_t> otherSize) {
        return allocate2D_internal<T>(otherSize);
    }

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
                .raw_length = sizeof(T) * size.x * size.y
        };

        T* data = malloc(stats.raw_length);
        allocatedPtrs.push_back(data);

        return Sim2DArray<T, MType::Cpu>(stats, data);
    }

    std::vector<void*> allocatedPtrs;
};

#if CUDA_ENABLED
template<>
class FrameAllocator<MType::Cuda> {
public:
    template<class T>
    Sim2DArray<T, MType::Cuda> allocate2D(Size<uint32_t> size) {
        return allocate2D_internal<T>(size);
    }

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
                .raw_length = sizeof(T) * size.x * size.y
        };

        T* data = nullptr;
        CHECKED_CUDA(cudaMalloc(&data, stats.raw_length));
        allocatedPtrs.push_back(data);

        return Sim2DArray<T, MType::Cuda>(stats, data);
    }

    std::vector<void*> allocatedPtrs;
};


/**
 * Allocator for Vulkan/Cuda shared memory.
 * Overall data for each frame is allocated all at once, then sub-allocated by allocate2D<T>().
 * This means the restrictions are tighter for TFrame - it must ONLY use VulkanCuda data for u,v,p,fluidmask.
 */
template<>
class FrameAllocator<MType::VulkanCuda> {
public:
    explicit FrameAllocator(VulkanContext& context, Size<uint32_t> paddedSize, size_t totalAllocationBytes)
            : paddedSize(paddedSize),
                    memory(context, totalAllocationBytes),
                    bytesUsed(0) {}

    Size<uint32_t> paddedSize;

    template<class T>
    Sim2DArray<T, MType::VulkanCuda> allocate2D(Size<uint32_t> size) {
        FATAL_ERROR_UNLESS(size != paddedSize, "FrameAllocator<VulkanCuda> expects all 2D allocations to be of size %d %d\n", paddedSize.x, paddedSize.y);
        return allocate2D_internal<T>(paddedSize);
    }

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::VulkanCuda, Storage> allocateRedBlack(Size<uint32_t> size) {
        FATAL_ERROR_UNLESS(size != paddedSize, "FrameAllocator<VulkanCuda> expects all 2D allocations to be of size %d %d\n", paddedSize.x, paddedSize.y);
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
                .raw_length = sizeof(T) * size.x * size.y
        };

        // Cast to char* to get correct pointer arithmetic, then cast back to T
        T* data = (T*)((char*)memory.as_cuda() + bytesUsed);

        vk::DescriptorBufferInfo vulkanBufferInfo;
        vulkanBufferInfo.buffer = memory.as_buffer();
        vulkanBufferInfo.offset = bytesUsed;
        vulkanBufferInfo.range = stats.raw_length;

        // Check if this allocation is actually valid
        bytesUsed += stats.raw_length;
        FATAL_ERROR_IF(bytesUsed > memory.sizeBytes, "FrameAllocator<Vulkan> out of memory");
        //TODO
//        FATAL_ERROR_IF(
//                (static_cast<std::uintptr_t>(data) % alignof(T)) == 0,
//                "FrameAllocator<Vulkan> allocated misaligned data pointer for %s", typeid(T).name()
//        );

        // Done - return the memory
        return Sim2DArray<T, MType::VulkanCuda>(stats, data, vulkanBufferInfo);
    }

    VulkanCudaBufferMemory memory;
    size_t bytesUsed;
};
#endif