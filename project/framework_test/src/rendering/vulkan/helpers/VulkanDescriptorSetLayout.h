//
// Created by samuel on 28/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>

class VulkanDescriptorSetLayout {
    vk::UniqueDescriptorSetLayout layout;

public:
    VulkanDescriptorSetLayout(vk::Device device, const std::vector<vk::DescriptorSetLayoutBinding>& bindings);
    VulkanDescriptorSetLayout(VulkanDescriptorSetLayout&&) noexcept = default;

    const vk::DescriptorSetLayout& operator*(){
        return *layout;
    }
};

