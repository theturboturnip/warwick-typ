//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/VulkanContext.h>

class VulkanFence {
    vk::UniqueFence fence;

public:
    explicit VulkanFence(VulkanContext& context, bool startSignalled=false);
    VulkanFence(VulkanFence&&) = default;

    const vk::Fence& operator *() const {
        return *fence;
    }
};

