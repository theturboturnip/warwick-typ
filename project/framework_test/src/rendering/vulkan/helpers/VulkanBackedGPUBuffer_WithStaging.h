//
// Created by samuel on 22/02/2021.
//

#pragma once

#include "VulkanBackedBuffer.h"

#include "util/ForgetOnMove.h"

/**
 * Helper class used for mapping memory in/out based on a struct's lifetime
 * i.e.
 *
 * {
 *      auto mappedMemory = buffer.mapMemory();
 *
 *      // Do things with memory[0], *memory etc.
 * } // lock is destroyed at the end of scope, memory is unmapped and flushed.
 */
class VulkanMappedMemory {
    vk::Device device;
    vk::DeviceMemory mappedMemory;
    ForgetOnMove<void*> data;

    VulkanMappedMemory(vk::Device device, vk::DeviceMemory mappedMemory);
    friend class VulkanBackedGPUBuffer_WithStaging;
public:
    VulkanMappedMemory(VulkanMappedMemory&&) = default;
    VulkanMappedMemory(const VulkanMappedMemory&) = delete;
    ~VulkanMappedMemory();

    void* operator*() {
        return data;
    }
};

/**
 * Class for a buffer that exists on the GPU, but can be mapped to the CPU and read.
 */
class VulkanBackedGPUBuffer_WithStaging {
    VulkanBackedBuffer gpuBuffer;
    VulkanBackedBuffer cpuStagingBuffer;

public:
    VulkanBackedGPUBuffer_WithStaging(VulkanContext& context, vk::BufferUsageFlags gpuUsage, size_t size, bool gpuShared=false);
    VulkanBackedGPUBuffer_WithStaging(VulkanBackedGPUBuffer_WithStaging&&) = default;
    VulkanBackedGPUBuffer_WithStaging(const VulkanBackedGPUBuffer_WithStaging&) = delete;

    VulkanMappedMemory mapCPUMemory(vk::Device device);

    void scheduleCopyToGPU(vk::CommandBuffer cmdBuffer);

    vk::DescriptorBufferInfo getGpuDescriptor() {
        return gpuBuffer.asDescriptor();
    }
    vk::Buffer getGpuBuffer() {
        return *gpuBuffer;
    }
};