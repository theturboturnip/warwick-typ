//
// Created by samuel on 19/02/2021.
//

#include "VulkanBackedBuffer.h"

VulkanBackedBuffer::VulkanBackedBuffer(VulkanContext& context, vk::MemoryPropertyFlags memoryProperties, vk::BufferUsageFlags usage, size_t size, bool shared)
    : size(size)
{
    {
        auto bufferCreateInfo = vk::BufferCreateInfo();
        bufferCreateInfo.flags = vk::BufferCreateFlagBits(0);
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = usage;

        auto queueFamilies = std::vector<uint32_t>({
            context.queueFamilies.computeFamily,
            context.queueFamilies.graphicsFamily
        });
        if (shared && (context.queueFamilies.computeFamily != context.queueFamilies.graphicsFamily)) {
            bufferCreateInfo.sharingMode = vk::SharingMode::eConcurrent;
            bufferCreateInfo.queueFamilyIndexCount = queueFamilies.size();
            bufferCreateInfo.pQueueFamilyIndices = queueFamilies.data();
        } else {
            bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        }

        buffer = context.device->createBufferUnique(bufferCreateInfo);
    }

    // Create device memory to back it
    {
        vk::MemoryRequirements memRequirements = context.device->getBufferMemoryRequirements(*buffer);

        bufferMemory = VulkanDeviceMemory(
                context,
                memRequirements,
                memoryProperties
        );
    }

// Bind the memory to the image
    context.device->bindBufferMemory(*buffer, *bufferMemory, 0);
}
