//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <optional>

struct VulkanQueueFamilies {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    static VulkanQueueFamilies fill_from_vulkan(vk::PhysicalDevice device, vk::UniqueSurfaceKHR& surface);

    [[nodiscard]] std::set<uint32_t> get_families() const {
        return {graphics_family.value(), present_family.value()};
    }

    [[nodiscard]] bool complete() const {
        return graphics_family.has_value() && present_family.has_value();
    }
};
