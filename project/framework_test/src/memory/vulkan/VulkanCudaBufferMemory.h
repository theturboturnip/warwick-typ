//
// Created by samuel on 09/02/2021.
//

#pragma once

#if CUDA_ENABLED
#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/helpers/VulkanDeviceMemory.h>
#include <rendering/vulkan/VulkanContext.h>
#include <cuda_runtime_api.h>

class VulkanCudaBufferMemory {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory deviceMemory;
    int cudaVulkanFd;
    cudaExternalMemory_t cudaExternalMemory = nullptr;
    void* cudaPointer = nullptr;

public:
    size_t sizeBytes;

    VulkanCudaBufferMemory(VulkanContext& context, size_t sizeBytes);
    VulkanCudaBufferMemory(VulkanCudaBufferMemory&&) noexcept = default;
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
#endif