//
// Created by samuel on 08/02/2021.
//

#include "VulkanFence.h"

VulkanFence::VulkanFence(VulkanContext& context, bool startSignalled) {
    vk::FenceCreateInfo info{};
    if (startSignalled) {
        info.flags = vk::FenceCreateFlagBits::eSignaled;
    }
    fence = context.device->createFenceUnique(info);
}
