//
// Created by samuel on 27/08/2020.
//

#include "CudaVulkan2DAllocator.cuh"

CudaVulkan2DAllocator::CudaVulkan2DAllocator(vk::Device device, vk::PhysicalDevice physicalDevice)
    : BaseVulkan2DAllocator(MemoryUsage::Device,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, // TODO - eHostVisible, eHostCached? if we try to read from the host?
                            device, physicalDevice),
      externalMemories()
{}

AllocatedMemory<void> CudaVulkan2DAllocator::mapFromVulkan_unsafe(BaseVulkan2DAllocator::VulkanMemory<void> vulkanMemory, size_t elemSize) {
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

    const size_t overallSize = vulkanMemory.unmappedMemoryInfo.totalSize * elemSize;

    cudaExternalMemoryHandleDesc memoryHandle{};
    memoryHandle.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryHandle.size = vulkanMemory.unmappedMemoryInfo.totalSize * elemSize;
    memoryHandle.handle.fd = fd;

    // TODO - Cuda Error Handling
    cudaExternalMemory_t cudaExternalMemory;
    cudaImportExternalMemory(&cudaExternalMemory, &memoryHandle);

    void* mappedBuffer = nullptr;
    cudaExternalMemoryBufferDesc buffer{};
    buffer.offset = 0;
    buffer.size = overallSize;
    buffer.flags = 0;

    cudaExternalMemoryGetMappedBuffer(&mappedBuffer, cudaExternalMemory, &buffer);

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
        cudaFree(externalMemory.cudaPtr);
        cudaDestroyExternalMemory(externalMemory.externalMemory);
    }
    externalMemories.clear();
    BaseVulkan2DAllocator::freeAll();
}
CudaVulkan2DAllocator::~CudaVulkan2DAllocator() {
    freeAll();
}
