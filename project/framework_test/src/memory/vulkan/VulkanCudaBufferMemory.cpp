//
// Created by samuel on 09/02/2021.
//

#include <util/fatal_error.h>
#include <util/check_vulkan_error.h>
#include "VulkanCudaBufferMemory.h"
#include <cuda_runtime_api.h>
#include <util/check_cuda_error.cuh>


VulkanCudaBufferMemory::VulkanCudaBufferMemory(VulkanContext& context, size_t sizeBytes) : sizeBytes(sizeBytes) {
    {
        vk::BufferCreateInfo bufferCreate{};
        bufferCreate.size = sizeBytes;
        bufferCreate.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        bufferCreate.sharingMode = vk::SharingMode::eExclusive;
        buffer = context.device->createBufferUnique(bufferCreate);
    }

    {
        // Create the device memory for the buffer.
        // Get the d-memory size/alignment required
        auto memoryRequirements = context.device->getBufferMemoryRequirements(*buffer);
        // Set it up to export to an opaque FD
        vk::ExportMemoryAllocateInfoKHR exportAllocInfo{};
        exportAllocInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd; // TODO - This won't work on windows. See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleVulkan/VulkanBaseApp.cpp#L1364
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = context.selectMemoryTypeIndex(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal);
        allocInfo.pNext = &exportAllocInfo;

        deviceMemory = context.device->allocateMemoryUnique(allocInfo);

        context.device->bindBufferMemory(*buffer, *deviceMemory, 0);
    }

    // Map it to CUDA
    {
        vk::MemoryGetFdInfoKHR vulkanMemoryExport{};
        vulkanMemoryExport.memory = *deviceMemory;
        vulkanMemoryExport.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

        CHECKED_VULKAN(context.dynamicLoader.vkGetMemoryFdKHR(*context.device, (VkMemoryGetFdInfoKHR*)&vulkanMemoryExport, &cudaVulkanFd));

        cudaExternalMemoryHandleDesc memoryHandle{};
        memoryHandle.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memoryHandle.size = sizeBytes;
        memoryHandle.handle.fd = cudaVulkanFd;

        // Import external memory into cudaExternalMemory
        CHECKED_CUDA(cudaImportExternalMemory(&cudaExternalMemory.get(), &memoryHandle));

        cudaExternalMemoryBufferDesc cudaBufferDesc{};
        cudaBufferDesc.offset = 0;
        cudaBufferDesc.size = sizeBytes;
        cudaBufferDesc.flags = 0;

        // Generate the pointer for cudaPointer
        CHECKED_CUDA(cudaExternalMemoryGetMappedBuffer(&cudaPointer.get(), cudaExternalMemory, &cudaBufferDesc));
    }
}

VulkanCudaBufferMemory::~VulkanCudaBufferMemory() {
    if (cudaPointer.has_value()) {
        CHECKED_CUDA(cudaFree(cudaPointer));
        CHECKED_CUDA(cudaDestroyExternalMemory(cudaExternalMemory));
    }
}
