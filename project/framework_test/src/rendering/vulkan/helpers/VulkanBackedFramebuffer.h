//
// Created by samuel on 11/02/2021.
//

#pragma once

#include "VulkanBackedGPUImage.h"
#include "VulkanFramebuffer.h"

class VulkanBackedFramebuffer {
    VulkanBackedGPUImage image;
    VulkanFramebuffer framebuffer;

public:
    VulkanBackedFramebuffer(VulkanContext& context, vk::ImageUsageFlags usage, Size<uint32_t> size, vk::RenderPass renderPass);
    VulkanBackedFramebuffer(VulkanBackedFramebuffer&&) noexcept = default;

    vk::Framebuffer operator *(){
        return *framebuffer;
    }
    vk::ImageView getImageView() {
        return framebuffer.getImageView();
    }
};
