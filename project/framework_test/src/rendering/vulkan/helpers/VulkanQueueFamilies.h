//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <optional>
#include <set>

struct VulkanQueueFamilies {
    uint32_t graphicsFamily;
    uint32_t presentFamily;

    static std::optional<VulkanQueueFamilies> getForDevice(vk::PhysicalDevice device, vk::UniqueSurfaceKHR& surface);

    [[nodiscard]] std::set<uint32_t> uniqueFamilies() const {
        return {graphicsFamily, presentFamily};
    }
};
