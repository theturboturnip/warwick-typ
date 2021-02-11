//
// Created by samuel on 11/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include "VulkanDeviceMemory.h"
#include "util/Size.h"

class VulkanBackedGPUImage {
    vk::UniqueImage image;
    VulkanDeviceMemory imageMemory;

public:
    vk::Format format;
    Size<uint32_t> size;

    VulkanBackedGPUImage(VulkanContext& context, vk::ImageUsageFlags usage, Size<uint32_t> size);
    VulkanBackedGPUImage(VulkanBackedGPUImage&&) noexcept = default;

    vk::DeviceMemory getMemory() {
        return *imageMemory;
    }
    vk::Image operator *(){
        return *image;
    }
};
