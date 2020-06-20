//
// Created by samuel on 18/06/2020.
//

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vulkan/vulkan.hpp>

#include "cuda_memory_wrappers.cuh"

template<typename T>
class CUDAVulkanSharedBuffer : CUDAUnified1DArray<T> {
public:
    CUDAVulkanSharedBuffer(vk::DeviceMemory& memory);

private:
    cudaExternalMemory_t cudaExtMemory;
};