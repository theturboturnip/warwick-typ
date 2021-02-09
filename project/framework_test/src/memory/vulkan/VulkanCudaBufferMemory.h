//
// Created by samuel on 09/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/helpers/VulkanDeviceMemory.h>
#include <rendering/vulkan/VulkanContext.h>
#include <cuda_runtime_api.h>

class VulkanCudaBufferMemory {
    vk::UniqueBuffer buffer;
    VulkanDeviceMemory deviceMemory;
    int cudaVulkanFd;
    cudaExternalMemory_t cudaExternalMemory;
    void* cudaPointer;

public:
    size_t sizeBytes;

    VulkanCudaBufferMemory(VulkanContext& context, size_t sizeBytes);
    VulkanCudaBufferMemory(VulkanCudaBufferMemory&&) = default;
    ~VulkanCudaBufferMemory();

    vk::Buffer as_buffer() {
        return *buffer;
    }
    vk::DeviceMemory as_deviceMemory() {
        return *deviceMemory;
    }
    void* as_cuda(){
        return cudaPointer;
    }
};

