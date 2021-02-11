//
// Created by samuel on 11/02/2021.
//

#include "VulkanBackedFramebuffer.h"

VulkanBackedFramebuffer::VulkanBackedFramebuffer(VulkanContext &context, vk::ImageUsageFlags usage,
                                                 Size<uint32_t> size, vk::RenderPass renderPass)
     : image(context, usage, size),
        framebuffer(context, *image, image.format, size, renderPass)
        {}
