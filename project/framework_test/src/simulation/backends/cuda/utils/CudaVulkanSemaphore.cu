//
// Created by samuel on 27/08/2020.
//

#include "CudaVulkanSemaphore.cuh"

#include <util/check_cuda_error.cuh>

CudaVulkanSemaphore::CudaVulkanSemaphore(vk::Device device, vk::Semaphore vulkanSemaphore)
    : vulkanSemaphore(vulkanSemaphore)
{
    int fd;

    vk::SemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
    semaphoreGetFdInfoKHR.pNext = nullptr;
    semaphoreGetFdInfoKHR.semaphore = vulkanSemaphore;
    semaphoreGetFdInfoKHR.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;

    PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
    fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!fpGetSemaphoreFdKHR) {
        throw std::runtime_error("Failed to retrieve vkGetSemaphoreFdKHR!");
    }
    if (fpGetSemaphoreFdKHR(device, &((VkSemaphoreGetFdInfoKHR&)semaphoreGetFdInfoKHR), &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }

    cudaExternalSemaphoreHandleDesc semaphoreHandleDesc{};
    semaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    semaphoreHandleDesc.handle.fd = fd;
    semaphoreHandleDesc.flags = 0;
    CHECKED_CUDA(cudaImportExternalSemaphore(&cudaSemaphore.get(), &semaphoreHandleDesc));
}
void CudaVulkanSemaphore::signalAsync(cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = 0;

    CHECKED_CUDA(cudaSignalExternalSemaphoresAsync(&cudaSemaphore.get(), &signalParams, 1, stream));
}
void CudaVulkanSemaphore::waitForAsync(cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = 0;

    CHECKED_CUDA(cudaWaitExternalSemaphoresAsync(&cudaSemaphore.get(), &waitParams, 1, stream));
}
CudaVulkanSemaphore::~CudaVulkanSemaphore() {
    if (cudaSemaphore.has_value()) {
        CHECKED_CUDA(cudaDestroyExternalSemaphore(cudaSemaphore.get()));
    }
}
