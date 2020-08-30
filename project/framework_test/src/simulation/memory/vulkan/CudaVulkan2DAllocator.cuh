//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "BaseVulkan2DAllocator.h"

#include <cuda_runtime_api.h>

class CudaVulkan2DAllocator : public BaseVulkan2DAllocator {
    struct CudaMappedMemory {
        cudaExternalMemory_t externalMemory;
        void* cudaPtr;
    };
    std::vector<CudaMappedMemory> externalMemories;

protected:
    AllocatedMemory<void> mapFromVulkan_unsafe(VulkanMemory<void>, size_t elemSize, const void* initialData) override;
public:
    CudaVulkan2DAllocator(vk::Device device, vk::PhysicalDevice physicalDevice);

    void freeAll() override;
    ~CudaVulkan2DAllocator() override;
};

