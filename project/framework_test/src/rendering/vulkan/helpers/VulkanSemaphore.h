//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <util/fatal_error.h>
#include <vulkan/vulkan.hpp>

class VulkanSemaphore {
    vk::UniqueSemaphore semaphore;

public:
    explicit VulkanSemaphore(vk::Device device);
    VulkanSemaphore(VulkanSemaphore&&) noexcept = default;

    const vk::Semaphore& operator *() const {
        return *semaphore;
    }
};

