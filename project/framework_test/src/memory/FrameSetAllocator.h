//
// Created by samuel on 09/02/2021.
//

#pragma once

#include "MType.h"
#include "Sim2DArray.h"
#include "FrameAllocator.h"

#include <cstdint>
#include <vector>

#include "util/Size.h"

template<MType MemType, class TFrame>
class FrameSetAllocator {
public:
    FrameSetAllocator(Size<uint32_t> paddedSize, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back(FrameAllocator<MemType>(paddedSize));
            frames.emplace_back(TFrame(paddedSize, frameAllocs[i]));
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
    static_assert(std::is_same_v<decltype(TFrame::u), Sim2DArray<float, MType::VulkanCuda>>);
    static_assert(std::is_same_v<decltype(TFrame::v), Sim2DArray<float, MType::VulkanCuda>>);
    static_assert(std::is_same_v<decltype(TFrame::p), Sim2DArray<float, MType::VulkanCuda>>);
    static_assert(std::is_same_v<decltype(TFrame::fluidmask), Sim2DArray<int, MType::VulkanCuda>>);

public:

    FrameSetAllocator(VulkanContext& context, Size<uint32_t> paddedSize, size_t frameCount) {
        frameAllocs.clear();
        frames.clear();
        vulkanFrames.clear();

        const size_t totalFrameSize = decltype(TFrame::u)::sizeBytesOf(paddedSize) +
                decltype(TFrame::v)::sizeBytesOf(paddedSize) +
                decltype(TFrame::p)::sizeBytesOf(paddedSize) +
                decltype(TFrame::fluidmask)::sizeBytesOf(paddedSize);

        for (size_t i = 0; i < frameCount; i++) {
            frameAllocs.emplace_back(FrameAllocator<MType::VulkanCuda>(context, paddedSize, totalFrameSize));
            frames.emplace_back(TFrame(paddedSize, frameAllocs[i]));
            const auto& frameRef = frames[i];
            vulkanFrames.emplace_back(VulkanSimFrameData{
                .matrixSize = paddedSize,
                .u = frameRef.u,
                .v = frameRef.v,
                .p = frameRef.p,
                .fluidmask = frameRef.fluidmask
            });
        }
    }

    std::vector<TFrame> frames;
    std::vector<VulkanSimFrameData> vulkanFrames;
protected:
    std::vector<FrameAllocator<MType::VulkanCuda>> frameAllocs;
};