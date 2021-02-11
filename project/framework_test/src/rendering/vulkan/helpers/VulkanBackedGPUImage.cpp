//
// Created by samuel on 11/02/2021.
//

#include "VulkanBackedGPUImage.h"

VulkanBackedGPUImage::VulkanBackedGPUImage(VulkanContext& context, vk::ImageUsageFlags usage, Size<uint32_t> size)
    : format(vk::Format::eR8G8B8A8Srgb),
      size(size)
{

    // Create VkImage
    {
        auto imageCreateInfo = vk::ImageCreateInfo{};
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        imageCreateInfo.usage = usage;//vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
        imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageCreateInfo.extent.width = size.x;
        imageCreateInfo.extent.height = size.y;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.format = format;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;

        image = context.device->createImageUnique(imageCreateInfo);
    }

    // Create device memory to back it
    {
        vk::MemoryRequirements memRequirements = context.device->getImageMemoryRequirements(*image);

        imageMemory = VulkanDeviceMemory(
                context,
                memRequirements,
                vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    }

    // Bind the memory to the image
    context.device->bindImageMemory(*image, *imageMemory, 0);
}
