//
// Created by samuel on 11/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/VulkanContext.h>
#include "util/Size.h"

class VulkanFramebuffer {
    vk::Image boundImage;
    vk::UniqueImageView imageView;
    vk::UniqueFramebuffer framebuffer;

public:
    VulkanFramebuffer(VulkanContext &context, vk::Image boundImage, vk::Format boundImageFormat, Size<uint32_t> boundImageSize, vk::RenderPass renderPass);
    VulkanFramebuffer(VulkanFramebuffer&&) noexcept = default;

    vk::Image getImage() {
        return boundImage;
    }
    vk::ImageView getImageView() {
        return *imageView;
    }
    vk::Framebuffer operator *(){
        return *framebuffer;
    }
};
