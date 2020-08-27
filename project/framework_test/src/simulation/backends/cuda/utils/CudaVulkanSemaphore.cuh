//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <cuda_runtime_api.h>
#include <util/fatal_error.h>
#include <vulkan/vulkan.hpp>

class CudaVulkanSemaphore {
    vk::Semaphore vulkanSemaphore;
    cudaExternalSemaphore_t cudaSemaphore = 0;

public:
    CudaVulkanSemaphore(vk::Device device, vk::Semaphore vulkanSemaphore);
    CudaVulkanSemaphore(const CudaVulkanSemaphore& copy) = delete;
    ~CudaVulkanSemaphore();

    void signalAsync(cudaStream_t);
    void waitForAsync(cudaStream_t);

    cudaExternalSemaphore_t operator*() const {
        return cudaSemaphore;
    }
};