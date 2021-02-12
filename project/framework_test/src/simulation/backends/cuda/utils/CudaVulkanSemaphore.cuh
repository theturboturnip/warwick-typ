//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <cuda_runtime_api.h>
#include <util/fatal_error.h>
#include <vulkan/vulkan.hpp>

class CudaVulkanSemaphore {
    vk::Semaphore vulkanSemaphore;
    cudaExternalSemaphore_t cudaSemaphore = nullptr;

public:
    CudaVulkanSemaphore(vk::Device device, vk::Semaphore vulkanSemaphore);
    CudaVulkanSemaphore(CudaVulkanSemaphore&&) noexcept = default;
    CudaVulkanSemaphore(const CudaVulkanSemaphore&) = delete;
    ~CudaVulkanSemaphore();

    void signalAsync(cudaStream_t);
    void waitForAsync(cudaStream_t);

    cudaExternalSemaphore_t operator*() const {
        return cudaSemaphore;
    }
};