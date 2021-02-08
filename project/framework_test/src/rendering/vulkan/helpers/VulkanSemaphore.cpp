//
// Created by samuel on 27/08/2020.
//

#include "VulkanSemaphore.h"

VulkanSemaphore::VulkanSemaphore(vk::Device device) {
    auto externalSemaphoreInfo = vk::ExportSemaphoreCreateInfo{};
    externalSemaphoreInfo.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;

    auto semaphoreInfo = vk::SemaphoreCreateInfo();
    semaphoreInfo.pNext = &externalSemaphoreInfo;

    semaphore = device.createSemaphoreUnique(semaphoreInfo);

//    if (startSignalled) {
//        vk::SemaphoreSignalInfo signalInfo{};
//        signalInfo.semaphore =
//
//        device.signalSemaphore();
//    }
}

