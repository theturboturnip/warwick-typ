//
// Created by samuel on 11/02/2021.
//

#include "VulkanFramebuffer.h"

VulkanFramebuffer::VulkanFramebuffer(VulkanContext &context, vk::Image boundImage, vk::Format boundImageFormat, Size<uint32_t> boundImageSize, vk::RenderPass renderPass) {
    // Create the image view
    {
        auto createInfo = vk::ImageViewCreateInfo();
        createInfo.image = boundImage;
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = boundImageFormat;

        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;

        createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        // We don't do any mipmapping/texture arrays ever - only use the first mip level, and the first array layer
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        imageView = context.device->createImageViewUnique(createInfo);
    }

    // Create framebuffer
    {
        auto framebufferCreateInfo = vk::FramebufferCreateInfo();
        framebufferCreateInfo.renderPass = renderPass;
        framebufferCreateInfo.attachmentCount = 1;
        framebufferCreateInfo.pAttachments = &(*imageView);
        framebufferCreateInfo.width = boundImageSize.x;
        framebufferCreateInfo.height = boundImageSize.y;
        framebufferCreateInfo.layers = 1;

        framebuffer = context.device->createFramebufferUnique(framebufferCreateInfo);
    }
}
