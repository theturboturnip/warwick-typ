//
// Created by samuel on 19/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include "VulkanDeviceMemory.h"

class VulkanBackedBuffer {
    vk::UniqueBuffer buffer;
    size_t size;
    VulkanDeviceMemory bufferMemory;

public:
    VulkanBackedBuffer(VulkanContext& context, vk::BufferUsageFlags usage, size_t size, bool shared=false);
    VulkanBackedBuffer(VulkanBackedBuffer&&) noexcept = default;
    VulkanBackedBuffer(const VulkanBackedBuffer&) = delete;

    vk::DescriptorBufferInfo asDescriptor() {
        return vk::DescriptorBufferInfo (*buffer, 0, size);
    }
    vk::DeviceMemory asDeviceMemory() {
        return *bufferMemory;
    }
    vk::Buffer operator*() {
        return *buffer;
    }
};

