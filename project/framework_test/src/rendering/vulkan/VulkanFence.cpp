//
// Created by samuel on 08/02/2021.
//

#include "VulkanFence.h"

VulkanFence::VulkanFence(vk::Device device) {
    vk::FenceCreateInfo info{};
    fence = device.createFenceUnique(info);
}
