//
// Created by samuel on 27/08/2020.
//

#include "CudaVulkan2DAllocator.cuh"
#include <util/check_cuda_error.cuh>

CudaVulkan2DAllocator::CudaVulkan2DAllocator(vk::Device device, vk::PhysicalDevice physicalDevice)
    : BaseVulkan2DAllocator(MemoryUsage::Device,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, // TODO - eHostVisible, eHostCached? if we try to read from the host?
                            device, physicalDevice),
      externalMemories()
{}

AllocatedMemory<void> CudaVulkan2DAllocator::mapFromVulkan_unsafe(BaseVulkan2DAllocator::VulkanMemory<void> vulkanMemory, size_t elemSize, const void* initialData) {
    int fd = -1;

    vk::MemoryGetFdInfoKHR vulkanMemoryExport{};
    vulkanMemoryExport.memory = vulkanMemory.deviceMemory;
    vulkanMemoryExport.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
    fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!fpGetMemoryFdKHR) {
        throw std::runtime_error("Failed to retrieve vkGetMemoryFdKHR!");
    }
    // TODO - it would be nice to use the vulkan.hpp dynamic dispatch here
    if (fpGetMemoryFdKHR(device, &(VkMemoryGetFdInfoKHR&)vulkanMemoryExport, &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }

    const size_t sizeBytes = vulkanMemory.unmappedMemoryInfo.totalSize * elemSize;

    cudaExternalMemoryHandleDesc memoryHandle{};
    memoryHandle.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryHandle.size = sizeBytes;
    memoryHandle.handle.fd = fd;

    cudaExternalMemory_t cudaExternalMemory;
    CHECKED_CUDA(cudaImportExternalMemory(&cudaExternalMemory, &memoryHandle));

    void* mappedBuffer = nullptr;
    cudaExternalMemoryBufferDesc buffer{};
    buffer.offset = 0;
    buffer.size = sizeBytes;
    buffer.flags = 0;

    CHECKED_CUDA(cudaExternalMemoryGetMappedBuffer(&mappedBuffer, cudaExternalMemory, &buffer));
    if (initialData)
        CHECKED_CUDA(cudaMemcpy(mappedBuffer, initialData, sizeBytes, cudaMemcpyDefault));
    else
        CHECKED_CUDA(cudaMemset(mappedBuffer, 0, sizeBytes));

    externalMemories.push_back(CudaMappedMemory{
            .externalMemory = cudaExternalMemory,
            .cudaPtr = mappedBuffer,
    });

    return AllocatedMemory<void>{
            .pointer = mappedBuffer,
            .totalSize = vulkanMemory.unmappedMemoryInfo.totalSize,
            .width = vulkanMemory.unmappedMemoryInfo.width,
            .height = vulkanMemory.unmappedMemoryInfo.height,
            .columnStride = vulkanMemory.unmappedMemoryInfo.columnStride,
    };
}
void CudaVulkan2DAllocator::freeAll() {
    for (CudaMappedMemory externalMemory : externalMemories) {
        CHECKED_CUDA(cudaFree(externalMemory.cudaPtr));
        CHECKED_CUDA(cudaDestroyExternalMemory(externalMemory.externalMemory));
    }
    externalMemories.clear();
    BaseVulkan2DAllocator::freeAll();
}
CudaVulkan2DAllocator::~CudaVulkan2DAllocator() {
    freeAll();
}
