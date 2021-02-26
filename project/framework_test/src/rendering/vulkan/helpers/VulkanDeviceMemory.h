//
// Created by samuel on 28/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/VulkanContext.h>

class VulkanDeviceMemory {
    vk::UniqueDeviceMemory memory;

public:
    // TODO - Make this consistent across the board - do we allow copy constructors or not?
    VulkanDeviceMemory() : memory(nullptr) {}
    VulkanDeviceMemory(VulkanContext& context, vk::MemoryRequirements requirements, vk::MemoryPropertyFlags properties);

    const vk::DeviceMemory& operator*() const {
        return *memory;
    }
};

