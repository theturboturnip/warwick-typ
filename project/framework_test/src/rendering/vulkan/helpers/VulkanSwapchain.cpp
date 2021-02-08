//
// Created by samuel on 08/02/2021.
//

#include "VulkanSwapchain.h"
#include <util/selectors.h>

VulkanSwapchain::VulkanSwapchain(VulkanContext& setup, VulkanRenderPass& swapchainRenderPass) {
    auto physicalDevice = setup.physicalDevice;
    auto logicalDevice = *setup.device;

    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*setup.surface);
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        // If the surface currently has an extent, just use that for the swapchain
        extents = surfaceCapabilities.currentExtent;
    } else {
        // The surface doesn't specify an extent to use, so select the one we want.
        // The tutorial just clamps the x/y inside the minimum/maximum ranges. If this ever happens everything is going to look weird, so we just stop.
        if (setup.windowSize.x < surfaceCapabilities.minImageExtent.width || surfaceCapabilities.maxImageExtent.width < setup.windowSize.x) {
            FATAL_ERROR("Window width %u out of range [%u, %u]\n", setup.windowSize.x, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
        }
        if (setup.windowSize.y < surfaceCapabilities.minImageExtent.height || surfaceCapabilities.maxImageExtent.height < setup.windowSize.y) {
            FATAL_ERROR("Window height %u out of range [%u, %u]\n", setup.windowSize.y, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
        }

        extents = vk::Extent2D(setup.windowSize.x, setup.windowSize.y);
    }

    // If we just took the minimum, we could end up having to wait on the driver before getting another image.
    // Get 1 extra, so 1 is always free at any given time
    imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 &&
        imageCount > surfaceCapabilities.maxImageCount) {
        // Make sure we don't exceed the maximum
        imageCount = surfaceCapabilities.maxImageCount;
    }

    auto swapchainCreateInfo = vk::SwapchainCreateInfoKHR();
    swapchainCreateInfo.surface = *setup.surface;

    swapchainCreateInfo.presentMode = setup.presentMode;
    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageExtent = extents;
    swapchainCreateInfo.imageFormat = setup.surfaceFormat.format;
    swapchainCreateInfo.imageColorSpace = setup.surfaceFormat.colorSpace;
    swapchainCreateInfo.imageArrayLayers = 1; // We're not rendering in stereoscopic 3D => set this to 1
    swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // Use eColorAttachment so we can directly render to the swapchain images.

    auto queueFamilyVector = std::vector<uint32_t>({setup.queueFamilies.graphicsFamily, setup.queueFamilies.presentFamily});
    if (setup.queueFamilies.graphicsFamily != setup.queueFamilies.presentFamily) {
        // The swapchain images need to be able to be used by both families.
        // Use Concurrent mode to make that possible.
        swapchainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        swapchainCreateInfo.queueFamilyIndexCount = queueFamilyVector.size();
        swapchainCreateInfo.pQueueFamilyIndices = queueFamilyVector.data();
    } else {
        // Same queue families => images can be owned exclusively by that queue family.
        // In this case we don't need to specify the different queues, because there is only one.
        swapchainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
        swapchainCreateInfo.queueFamilyIndexCount = 0;
        swapchainCreateInfo.pQueueFamilyIndices = nullptr;
    }

    // Extra stuff
    // Set the swapchain rotation to the current rotation of the surface
    swapchainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
    // Don't apply the alpha channel as transparency to the window
    // i.e. if a pixel has alpha = 0 in the presented image it will be opqaue in the window system
    swapchainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = nullptr;

    swapchain = logicalDevice.createSwapchainKHRUnique(swapchainCreateInfo);
    images = logicalDevice.getSwapchainImagesKHR(*swapchain);

    // Make image views
    auto makeIdentityView = [&setup](vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectFlags = vk::ImageAspectFlagBits::eColor){
        auto createInfo = vk::ImageViewCreateInfo();
        createInfo.image = image;
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = format;

        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;

        createInfo.subresourceRange.aspectMask = aspectFlags;
        // We don't do any mipmapping/texture arrays ever - only use the first mip level, and the first array layer
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        return setup.device->createImageViewUnique(createInfo);
    };
    imageViews.clear();
    for (vk::Image image : images) {
        imageViews.push_back(makeIdentityView(image, setup.surfaceFormat.format));
    }

    FATAL_ERROR_UNLESS(swapchainRenderPass.colorAttachmentFormat == setup.surfaceFormat.format, "Render pass using for render to swapchain doesn't render to correct format.")

    // Make framebuffers
    {
        framebuffers.clear();
        for (const auto& imageView : imageViews) {
            auto framebufferCreateInfo = vk::FramebufferCreateInfo();
            framebufferCreateInfo.renderPass = *swapchainRenderPass;
            framebufferCreateInfo.attachmentCount = 1;
            framebufferCreateInfo.pAttachments = &(*imageView);
            framebufferCreateInfo.width = setup.windowSize.x;
            framebufferCreateInfo.height = setup.windowSize.y;
            framebufferCreateInfo.layers = 1;

            framebuffers.push_back(logicalDevice.createFramebufferUnique(framebufferCreateInfo));
        }
    }
}
