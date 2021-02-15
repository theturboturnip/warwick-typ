//
// Created by samuel on 13/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/VulkanContext.h>
#include "VulkanBackedGPUImage.h"

class VulkanImageSampler {
    // TODO - could make these private and expose through getters.
public:
    vk::UniqueImageView imageView;
    vk::UniqueSampler sampler;

    VulkanImageSampler(vk::Device device, vk::Image image, vk::Format format);
    VulkanImageSampler(VulkanContext& context, VulkanBackedGPUImage& image);
    VulkanImageSampler(VulkanImageSampler&&) noexcept = default;
    VulkanImageSampler(const VulkanImageSampler&) = delete;
};

