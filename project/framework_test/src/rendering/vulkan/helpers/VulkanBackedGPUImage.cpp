//
// Created by samuel on 11/02/2021.
//

#include "VulkanBackedGPUImage.h"

VulkanBackedGPUImage::VulkanBackedGPUImage(VulkanContext& context, vk::ImageUsageFlags usage, Size<uint32_t> size, vk::Format format, bool shared)
    : format(format),
      size(size)
{

    // Create VkImage
    {
        // Queue families this would be shared between, if it was shared.
        // Only used if shared and if necessary.
        // TODO - image sharing may be unnecessary if ownership can be transferred
        //  https://harrylovescode.gitbooks.io/vulkan-api/content/chap07/chap07.html
        uint32_t sharedQueueFamilies[] = {
                context.queueFamilies.computeFamily,
                context.queueFamilies.graphicsFamily
        };

        auto imageCreateInfo = vk::ImageCreateInfo{};
        imageCreateInfo.imageType = vk::ImageType::e2D;
        if (shared && (context.queueFamilies.computeFamily != context.queueFamilies.graphicsFamily)) {
            imageCreateInfo.sharingMode = vk::SharingMode::eConcurrent;
            imageCreateInfo.queueFamilyIndexCount = 2;
            imageCreateInfo.pQueueFamilyIndices = sharedQueueFamilies;
        } else {
            imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        }
        imageCreateInfo.usage = usage;
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
