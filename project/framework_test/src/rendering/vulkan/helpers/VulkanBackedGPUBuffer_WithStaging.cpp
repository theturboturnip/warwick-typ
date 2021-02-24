//
// Created by samuel on 22/02/2021.
//

#include "VulkanBackedGPUBuffer_WithStaging.h"

VulkanMappedMemory::VulkanMappedMemory(vk::Device device, vk::DeviceMemory mappedMemory)
    : device(device),
      mappedMemory(mappedMemory),
      data(device.mapMemory(mappedMemory, 0, VK_WHOLE_SIZE)) {}

VulkanMappedMemory::~VulkanMappedMemory() {
    if (data.has_value()) {
//        auto memoryRange = vk::MappedMemoryRange(
//                mappedMemory,
//                0,
//                VK_WHOLE_SIZE
//        );
//        device.flushMappedMemoryRanges({memoryRange});
        device.unmapMemory(mappedMemory);
        data.release();
    }
}

VulkanBackedGPUBuffer_WithStaging::VulkanBackedGPUBuffer_WithStaging(
        VulkanContext &context,
        vk::BufferUsageFlags gpuUsage,
        size_t size, bool gpuShared)
    : gpuBuffer(context, vk::MemoryPropertyFlagBits::eDeviceLocal, gpuUsage | vk::BufferUsageFlagBits::eTransferDst, size, gpuShared),
      cpuStagingBuffer(context,
                       // Map host-visible (CPU can see), host coherent (CPU can write without having to flush afterwards)
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                       // Just a transfer source, not used in any other way
                       vk::BufferUsageFlagBits::eTransferSrc,
                       size,
                       // Make it shared so it can be used for copies in both the Graphics and Compute command buffers.
                       true){}

VulkanMappedMemory VulkanBackedGPUBuffer_WithStaging::mapCPUMemory(vk::Device device) {
    return VulkanMappedMemory(device, cpuStagingBuffer.asDeviceMemory());
}

void VulkanBackedGPUBuffer_WithStaging::scheduleCopyToGPU(vk::CommandBuffer cmdBuffer) {
    auto copyRegion = vk::BufferCopy{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = gpuBuffer.size; // Equal to CPU size
    cmdBuffer.copyBuffer(*cpuStagingBuffer, *gpuBuffer, {copyRegion});
}
