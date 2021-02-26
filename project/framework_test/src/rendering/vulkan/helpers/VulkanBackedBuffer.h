//
// Created by samuel on 19/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include "VulkanDeviceMemory.h"

class VulkanBackedBuffer {
    vk::UniqueBuffer buffer;
    VulkanDeviceMemory bufferMemory;

public:
    size_t size;

    VulkanBackedBuffer(VulkanContext& context, vk::MemoryPropertyFlags memoryProperties, vk::BufferUsageFlags usage, size_t size, bool shared=false);
    VulkanBackedBuffer(VulkanBackedBuffer&&) noexcept = default;
    VulkanBackedBuffer(const VulkanBackedBuffer&) = delete;

    [[nodiscard]] vk::DescriptorBufferInfo asDescriptor() const {
        return vk::DescriptorBufferInfo (*buffer, 0, size);
    }
    [[nodiscard]] vk::DeviceMemory asDeviceMemory() const {
        return *bufferMemory;
    }
    vk::Buffer operator*() const {
        return *buffer;
    }
};

