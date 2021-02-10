//
// Created by samuel on 09/02/2021.
//

#pragma once

#include "memory/internal/MType.h"
#include "memory/internal/Sim2DArray.h"
#include "FrameAllocator.h"

#include <cstdint>
#include <vector>
#include <simulation/file_format/SimSnapshot.h>

#include "util/Size.h"

template<MType MemType, class TFrame>
class FrameSetAllocator {
public:
    FrameSetAllocator(const SimSnapshot& s, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back(FrameAllocator<MemType>());
            frames.emplace_back(TFrame(frameAllocs[i]));
        }
    }

    std::vector<TFrame> frames;
protected:
    std::vector<FrameAllocator<MemType>> frameAllocs;
};

struct VulkanSimFrameData {
    Size<uint32_t> matrixSize = {0, 0};

    vk::DescriptorBufferInfo u;
    vk::DescriptorBufferInfo v;
    vk::DescriptorBufferInfo p;
    vk::DescriptorBufferInfo fluidmask;
};

template<class TFrame>
class FrameSetAllocator<MType::VulkanCuda, TFrame> {
    // Check TFrame is a correct Vulkan-enabled Frame
    static_assert(decltype(TFrame::u)::MemType == MType::VulkanCuda);
    static_assert(decltype(TFrame::v)::MemType == MType::VulkanCuda);
    static_assert(decltype(TFrame::p)::MemType == MType::VulkanCuda);
    static_assert(decltype(TFrame::fluidmask)::MemType == MType::VulkanCuda);

public:

    FrameSetAllocator(VulkanContext& context, Size<uint32_t> paddedSize, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();
        vulkanFrames.clear();

        const size_t totalFrameSizeBytes = decltype(TFrame::u)::sizeBytesOf(paddedSize) +
                decltype(TFrame::v)::sizeBytesOf(paddedSize) +
                decltype(TFrame::p)::sizeBytesOf(paddedSize) +
                decltype(TFrame::fluidmask)::sizeBytesOf(paddedSize);

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back(FrameAllocator<MType::VulkanCuda>(context, paddedSize, totalFrameSizeBytes));
            frames.emplace_back(TFrame(frameAllocs[i], cudaAlloc, paddedSize));
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

    std::vector<TFrame> frames;
    std::vector<VulkanSimFrameData> vulkanFrames;
protected:
    std::vector<FrameAllocator<MType::VulkanCuda>> frameAllocs;
    FrameAllocator<MType::Cuda> cudaAlloc;
};