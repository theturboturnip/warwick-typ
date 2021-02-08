//
// Created by samuel on 28/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>

class VulkanDeviceMemory {
    vk::UniqueDeviceMemory memory;

    static uint32_t selectMemoryHeap(vk::PhysicalDevice physicalDevice, uint32_t memoryTypeBits, vk::MemoryPropertyFlags properties);

public:
    // TODO - Make this consistent across the board - do we allow copy constructors or not?
    VulkanDeviceMemory() : memory(nullptr) {}
    VulkanDeviceMemory(vk::Device device, vk::PhysicalDevice physicalDevice, vk::MemoryRequirements requirements, vk::MemoryPropertyFlags properties);

    const vk::DeviceMemory& operator*() {
        return *memory;
    }
};

