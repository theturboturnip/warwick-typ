//
// Created by samuel on 13/02/2021.
//

#include "VulkanImageSampler.h"

VulkanImageSampler::VulkanImageSampler(vk::Device device, vk::Image image, vk::Format format) {
    // Create the image view
    {
        auto createInfo = vk::ImageViewCreateInfo();
        createInfo.image = image;
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = format;

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

        imageView = device.createImageViewUnique(createInfo);
    }

    // Create the sampler
    {
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.magFilter = vk::Filter::eLinear;

        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToBorder;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToBorder;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;

        samplerInfo.anisotropyEnable = false;
        samplerInfo.unnormalizedCoordinates = false;
        samplerInfo.compareEnable = false;

        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
        samplerInfo.mipLodBias = 0;
        samplerInfo.minLod = 0;
        samplerInfo.maxLod = 0;

        sampler = device.createSamplerUnique(samplerInfo);
    }
}

VulkanImageSampler::VulkanImageSampler(VulkanContext &context, VulkanBackedGPUImage &image)
    : VulkanImageSampler(*context.device, *image, image.format) {}
