//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <optional>

struct VulkanQueueFamilies {
    std::optional<uint32_t> graphics_family;

    static VulkanQueueFamilies fill_from_vulkan(vk::PhysicalDevice device);

    [[nodiscard]] bool complete() {
        return graphics_family.has_value();
    }
};
