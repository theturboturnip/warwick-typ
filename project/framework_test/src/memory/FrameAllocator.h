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
#include "util/check_cuda_error.cuh"
#endif

/*class BaseFrameAllocator {
public:
    Size<uint32_t> paddedSize;

protected:
    explicit BaseFrameAllocator(Size<uint32_t> paddedSize) : paddedSize(paddedSize){}

    template <class T, MType MemType, RedBlackStorage Storage>
    SimRedBlackArray<T, MemType, Storage> allocateRedBlack() {
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D<T>(splitSize);
        auto black = allocate2D<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D<T>(paddedSize);

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
};*/

/**
 * FrameAllocator<MType> exposes an interface for allocating memory of certain types.
 * allocate2D_mtype<T>() returns a Sim2DArray<T> for the specific memory type, of the correct frame size.
 * allocateRedBlack_mtype<T, RedBlackStorage> returns a SimRedBlackArray<T, Storage> for the specific memory type of the correct size.
 */

template<>
class FrameAllocator<MType::Cpu> {
public:
    explicit FrameAllocator(Size<uint32_t> paddedSize)
        : paddedSize(paddedSize) {}

    Size<uint32_t> paddedSize;

    template<class T>
    Sim2DArray<T, MType::Cpu> allocate2D_cpu() {
        allocate2D<T>(paddedSize);
    }

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::Cpu, Storage> allocateRedBlack_cpu() {
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D<T>(splitSize);
        auto black = allocate2D<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D<T>(paddedSize);

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
    Sim2DArray<T, MType::Cpu> allocate2D(Size<uint32_t> size) {
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
    explicit FrameAllocator(Size<uint32_t> paddedSize)
            : paddedSize(paddedSize) {}

    Size<uint32_t> paddedSize;

    template<class T>
    Sim2DArray<T, MType::Cuda> allocate2D_cuda() {
        allocate2D<T>(paddedSize);
    }

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::Cuda, Storage> allocateRedBlack_cuda() {
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D<T>(splitSize);
        auto black = allocate2D<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D<T>(paddedSize);

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
    }

    ~FrameAllocator() {
        for (void* ptr : allocatedPtrs) {
            cudaFree(ptr);
        }
        allocatedPtrs.clear();
    }

private:
    template<class T>
    Sim2DArray<T, MType::Cuda> allocate2D(Size<uint32_t> size) {
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

// Allocator<VulkanCuda> inherits from Allocator<Cuda> so that it can still allocate unified memory if requested


/**
 * Allocator for Vulkan/Cuda shared memory.
 * Overall data for each frame is allocated all at once, then sub-allocated by allocate2D<T>().
 * This means the restrictions are tighter for TFrame - it must ONLY use VulkanCuda data for u,v,p,fluidmask.
 */
template<>
class FrameAllocator<MType::VulkanCuda> : FrameAllocator<MType::Cuda> {
public:
    // INCOMPLETE
    static_assert(false, "Incomplete");

    // TODO - allocate device memory
    explicit FrameAllocator(Size<uint32_t> paddedSize)
            : FrameAllocator<MType::Cuda>(paddedSize) {}

    template<class T>
    Sim2DArray<T, MType::VulkanCuda> allocate2D_vkcuda() {
        allocate2D<T>(paddedSize);
    }

    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::VulkanCuda, Storage> allocateRedBlack_vkcuda() {
        auto splitSize = Size<uint32_t>(paddedSize.x, paddedSize.y/2);
        auto red = allocate2D<T>(splitSize);
        auto black = allocate2D<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D<T>(paddedSize);

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
    }


    ~FrameAllocator() {
        // TODO - free device memory
    }

private:
    template<class T>
    Sim2DArray<T, MType::VulkanCuda> allocate2D(Size<uint32_t> size) {

        Sim2DArrayStats stats = {
                .width = size.x,
                .height = size.y,
                .col_pitch = size.y,
                .raw_length = sizeof(T) * size.x * size.y
        };

        T* data = ;// TODO

        return Sim2DArray<T, MType::VulkanCuda>(stats, data);
    }
};
#endif