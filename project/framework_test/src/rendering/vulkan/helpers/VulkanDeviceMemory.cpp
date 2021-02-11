//
// Created by samuel on 28/08/2020.
//

#include "VulkanDeviceMemory.h"

VulkanDeviceMemory::VulkanDeviceMemory(VulkanContext& context, vk::MemoryRequirements requirements, vk::MemoryPropertyFlags properties)
    : memory(context.device->allocateMemoryUnique(
            vk::MemoryAllocateInfo(
                    requirements.size,
                    context.selectMemoryTypeIndex(requirements, properties)
            )
    )){
}
