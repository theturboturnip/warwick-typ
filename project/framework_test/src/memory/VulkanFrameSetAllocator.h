//
// Created by samuel on 10/02/2021.
//

#pragma once

#include "util/Size.h"
#include <vulkan/vulkan.hpp>

struct VulkanSimFrameData {
    Size<uint32_t> matrixSize = {0, 0};

    vk::DescriptorBufferInfo u;
    vk::DescriptorBufferInfo v;
    vk::DescriptorBufferInfo p;
    vk::DescriptorBufferInfo fluidmask;
};

class VulkanFrameSetAllocator {
public:
    virtual ~VulkanFrameSetAllocator() = default;

    std::vector<VulkanSimFrameData> vulkanFrames;
};