//
// Created by samuel on 10/02/2021.
//

#pragma once

#include "util/Size.h"
#include <vulkan/vulkan.hpp>

/**
 * This defines base structures for the VulkanFrameSetAllocator virtual base class.
 * These structures are used by a Vulkan renderer looking to import memory from a separate, potentially not-Vulkan-based simulation backend.
 */

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