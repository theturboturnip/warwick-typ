//
// Created by samuel on 27/08/2020.
//

#include "BaseVulkan2DAllocator.h"
BaseVulkan2DAllocator::BaseVulkan2DAllocator(const uint32_t usage, const vk::MemoryPropertyFlags expectedMemoryFlags, vk::Device device, vk::PhysicalDevice physicalDevice)
    : I2DAllocator(usage), memProperties(physicalDevice.getMemoryProperties()), expectedMemoryFlags(expectedMemoryFlags), device(device)
{}

uint32_t BaseVulkan2DAllocator::selectMemoryTypeIndex(uint32_t memoryTypeBits) {
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & expectedMemoryFlags) == expectedMemoryFlags) {
            return i;
        }
    }
    FATAL_ERROR("Couldn't find suitable memory type!");
}

BaseVulkan2DAllocator::VulkanMemory<void> BaseVulkan2DAllocator::allocateVulkan_unsafe(Size<uint32_t> size, size_t elemSize) {
    const size_t sizeBytes = size.x * size.y * elemSize;

    // The buffer is only used by the graphics queue
    vk::BufferCreateInfo bufferCreate{};
    bufferCreate.size = sizeBytes;
    bufferCreate.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    bufferCreate.sharingMode = vk::SharingMode::eExclusive;
    vk::UniqueBuffer buffer = device.createBufferUnique(bufferCreate);

    auto memoryRequirements = device.getBufferMemoryRequirements(*buffer);


    vk::ExportMemoryAllocateInfoKHR exportAllocInfo{};
    exportAllocInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd; // TODO - This won't work on windows. See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleVulkan/VulkanBaseApp.cpp#L1364
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = selectMemoryTypeIndex(memoryRequirements.memoryTypeBits);
    allocInfo.pNext = &exportAllocInfo;

    vk::UniqueDeviceMemory deviceMem = device.allocateMemoryUnique(allocInfo);

    device.bindBufferMemory(*buffer, *deviceMem, 0);

    // TODO - NAUGHTY!

    // Create the toReturn before moving the deviceMem and buffer into the memories list.
    const auto toReturn = VulkanMemory<void>{
            .deviceMemory = *deviceMem,
            .buffer = *buffer,
            .unmappedMemoryInfo = {
                    .pointer = nullptr,
                    .totalSize = size.x * size.y,

                    .width = size.x,
                    .height = size.y,
                    .columnStride = size.y,
            }
    };

    memories.push_back(VulkanOwnedMemory{
            .deviceMemory = std::move(deviceMem),
            .buffer = std::move(buffer)
    });

    return toReturn;
}
void BaseVulkan2DAllocator::freeAll() {
    memories.clear();
}
BaseVulkan2DAllocator::~BaseVulkan2DAllocator() {
    freeAll();
}
