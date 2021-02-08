//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vulkan/vulkan.hpp>

class VulkanFence {
    vk::UniqueFence fence;

public:
    explicit VulkanFence(vk::Device device);
    VulkanFence(VulkanFence&&) = default;

    const vk::Fence& operator *() const {
        return *fence;
    }
};

