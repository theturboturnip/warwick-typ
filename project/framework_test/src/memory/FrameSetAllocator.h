//
// Created by samuel on 09/02/2021.
//

#pragma once

#include "memory/internal/MType.h"
#include "memory/internal/Sim2DArray.h"
#include "FrameAllocator.h"
#include "VulkanFrameSetAllocator.h"

#include <cstdint>
#include <vector>
#include <simulation/file_format/SimSnapshot.h>

#include "util/Size.h"

/**
 * This file defines a set of FrameSetAllocator template specializations.
 * These allocate memory for a set of frames by creating separate FrameAllocator instances for each one.
 * The FrameAllocator instances are owned by the FrameSetAllocator, so all memory is also owned by the FrameSetAllocator.
 * This means FrameSetAllocator cannot be copied, only moved.
 */

template<MType MemType, class TFrame>
class FrameSetAllocator {
public:
    FrameSetAllocator(const SimSnapshot& s, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();
        frameAllocs.reserve(frameCount);

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back();
            frames.emplace_back(TFrame(frameAllocs[i], s.simSize.padded_pixel_size));
        }
    }
    FrameSetAllocator(FrameSetAllocator&&) noexcept = default;
    FrameSetAllocator(const FrameSetAllocator&) = delete;

    std::vector<TFrame> frames;
protected:
    std::vector<FrameAllocator<MemType>> frameAllocs;
};

#if CUDA_ENABLED
template<class TFrame>
class FrameSetAllocator<MType::Cuda, TFrame> {
public:
    FrameSetAllocator(const SimSnapshot& s, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();
        frameAllocs.reserve(frameCount);

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back();
            frames.emplace_back(TFrame(frameAllocs[i], frameAllocs[i], s.simSize.padded_pixel_size));
        }
    }
    FrameSetAllocator(FrameSetAllocator&&) noexcept = default;
    FrameSetAllocator(const FrameSetAllocator&) = delete;

    std::vector<TFrame> frames;
protected:
    std::vector<FrameAllocator<MType::Cuda>> frameAllocs;
};

/**
 * This allocates a set of Vulkan/Cuda frames.
 * These frames have extra restrictions:
 * They must take both a Vulkan and Cuda FrameAllocator in their constructor, so they can allocate CUDA managed memory separately.
 * They must have u,v,p, and fluidmask fields that all use VulkanCuda memory.
 *  This is so that the data can be extracted and shared with a potential Vulkan renderer.
 */
template<class TFrame>
class FrameSetAllocator<MType::VulkanCuda, TFrame> : public VulkanFrameSetAllocator {
    // Check TFrame is a correct Vulkan-enabled Frame
    static_assert(decltype(TFrame::u)::MemType == MType::VulkanCuda);
    static_assert(decltype(TFrame::v)::MemType == MType::VulkanCuda);
    static_assert(decltype(TFrame::p)::MemType == MType::VulkanCuda);
    static_assert(decltype(TFrame::fluidmask)::MemType == MType::VulkanCuda);

public:

    FrameSetAllocator(VulkanContext& context, Size<uint32_t> paddedSize, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();
        frameAllocs.reserve(frameCount);
        vulkanFrames.clear();
        vulkanFrames.reserve(frameCount);

        const size_t totalFrameSizeBytes = decltype(TFrame::u)::sizeBytesOf(paddedSize) +
                decltype(TFrame::v)::sizeBytesOf(paddedSize) +
                decltype(TFrame::p)::sizeBytesOf(paddedSize) +
                decltype(TFrame::fluidmask)::sizeBytesOf(paddedSize);

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back(context, paddedSize, totalFrameSizeBytes);
            frames.emplace_back(frameAllocs[i], cudaAlloc, paddedSize);
            const auto& frameRef = frames[i];
            vulkanFrames.emplace_back(VulkanSimFrameData{
                .matrixSize = paddedSize,
                .u = frameRef.u.as_vulkan(),
                .v = frameRef.v.as_vulkan(),
                .p = frameRef.p.joined.as_vulkan(),
                .fluidmask = frameRef.fluidmask.as_vulkan()
            });
        }
    }
    FrameSetAllocator(FrameSetAllocator&&) noexcept = default;
    FrameSetAllocator(const FrameSetAllocator&) = delete;

    ~FrameSetAllocator() override = default;

    std::vector<TFrame> frames;
protected:
    std::vector<FrameAllocator<MType::VulkanCuda>> frameAllocs;
    FrameAllocator<MType::Cuda> cudaAlloc;
};
#endif