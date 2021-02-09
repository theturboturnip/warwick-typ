//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "MType.h"
#include "Allocator_fwd.h" // To ensure we match the fwd decl.
#include "Sim2DArray.h"
#include "SimRedBlackArray.h"

#include "util/Size.h"

#if CUDA_ENABLED
#include <cuda_runtime_api.h>
#include "util/check_cuda_error.cuh"
#endif

template<>
class Allocator<MType::Cpu> {
public:
    template<class T>
    Sim2DArray<T, MType::Cpu> allocate2D(Size<uint32_t> size) {
        Sim2DArrayStats stats = {
                .width = size.x,
                .height = size.y,
                .col_pitch = size.y,
                .raw_length = sizeof(T) * size.x * size.y
        };

        T* data = malloc(stats.raw_length);

        return Sim2DArray<T, MType::Cpu>(stats, data);
    }
    template<class T, RedBlackStorage Storage>
    SimRedBlackArray<T, MType::Cpu, Storage> allocateRedBlack(Size<uint32_t> joinedSize) {
        auto splitSize = Size<uint32_t>(joinedSize.x, joinedSize.y/2);
        auto red = allocate2D<T>(splitSize);
        auto black = allocate2D<T>(splitSize);

        if constexpr (Storage == RedBlackStorage::WithJoined) {
            auto joined = allocate2D<T>(joinedSize);

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

    template<class TFrame>
    std::vector<TFrame> allocateFrames(size_t frameCount) {
        std::vector<TFrame> frames{};

        for (size_t i = 0; i < frameCount; i++) {
            frames.emplace_back(TFrame(*this));
        }

        return frames;
    }
};

#if CUDA_ENABLED
template<>
class Allocator<MType::Cuda> {
public:
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

        return Sim2DArray<T, MType::Cuda>(stats, data);
    }

    template<class TFrame>
    std::vector<TFrame> allocateFrames(size_t frameCount) {
        std::vector<TFrame> frames{};

        for (size_t i = 0; i < frameCount; i++) {
            frames.emplace_back(TFrame(*this));
        }

        return frames;
    }
};

// Allocator<VulkanCuda> inherits from Allocator<Cuda> so that it can still allocate unified memory if requested
struct VulkanFrame {
    Size<uint32_t> matrixSize = {0, 0};

    vk::DescriptorBufferInfo u;
    vk::DescriptorBufferInfo v;
    vk::DescriptorBufferInfo p;
    vk::DescriptorBufferInfo fluidmask;
};

template<>
class Allocator<MType::VulkanCuda> : Allocator<MType::Cuda> {
public:
    std::vector<VulkanFrame> vulkanFrames;

    // INCOMPLETE
    static_assert(false, "Incomplete");

    template<class T>
    Sim2DArray<T, MType::VulkanCuda> allocate2D(Size<uint32_t> size) {
        Sim2DArrayStats stats = {
                .width = size.x,
                .height = size.y,
                .col_pitch = size.y,
                .raw_length = sizeof(T) * size.x * size.y
        };

        T* data = cudaMalloc(stats.raw_length);

        return Sim2DArray<T, MType::VulkanCuda>(stats, data);
    }
    // TODO - redblack arrays

    // TODO - add paddedSize to all other things
    template<class TFrame>
    std::vector<TFrame> allocateFrames(Size<uint32_t> paddedSize, size_t frameCount) {
        std::vector<TFrame> frames{};
        vulkanFrames.clear();

        for (size_t i = 0; i < frameCount; i++) {
            frames.emplace_back(TFrame(*this, paddedSize));
        }

        return frames;
    }
};
#endif