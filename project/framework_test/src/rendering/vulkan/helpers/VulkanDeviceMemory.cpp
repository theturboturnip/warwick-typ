//
// Created by samuel on 28/08/2020.
//

#include "VulkanDeviceMemory.h"
#include <util/fatal_error.h>

VulkanDeviceMemory::VulkanDeviceMemory(vk::Device device, vk::PhysicalDevice physicalDevice, vk::MemoryRequirements requirements, vk::MemoryPropertyFlags properties) {
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = selectMemoryHeap(physicalDevice, requirements.memoryTypeBits, properties);// findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memory = device.allocateMemoryUnique(allocInfo);
}

uint32_t VulkanDeviceMemory::selectMemoryHeap(vk::PhysicalDevice physicalDevice, uint32_t memoryTypeBits, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    FATAL_ERROR("Couldn't find suitable memory type!");
}
