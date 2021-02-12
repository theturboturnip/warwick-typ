//
// Created by samuel on 08/02/2021.
//

#include "VulkanSwapchain.h"
#include <util/selectors.h>

VulkanSwapchain::VulkanSwapchain(VulkanContext& context, VulkanRenderPass& swapchainRenderPass) {
    auto physicalDevice = context.physicalDevice;
    auto logicalDevice = *context.device;

    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*context.surface);
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        // If the surface currently has an extent, just use that for the swapchain
        extents = surfaceCapabilities.currentExtent;
    } else {
        // The surface doesn't specify an extent to use, so select the one we want.
        // The tutorial just clamps the x/y inside the minimum/maximum ranges. If this ever happens everything is going to look weird, so we just stop.
        if (context.windowSize.x < surfaceCapabilities.minImageExtent.width || surfaceCapabilities.maxImageExtent.width < context.windowSize.x) {
            FATAL_ERROR("Window width %u out of range [%u, %u]\n", context.windowSize.x, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
        }
        if (context.windowSize.y < surfaceCapabilities.minImageExtent.height || surfaceCapabilities.maxImageExtent.height < context.windowSize.y) {
            FATAL_ERROR("Window height %u out of range [%u, %u]\n", context.windowSize.y, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
        }

        extents = vk::Extent2D(context.windowSize.x, context.windowSize.y);
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
    swapchainCreateInfo.surface = *context.surface;

    swapchainCreateInfo.presentMode = context.presentMode;
    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageExtent = extents;
    swapchainCreateInfo.imageFormat = context.surfaceFormat.format;
    swapchainCreateInfo.imageColorSpace = context.surfaceFormat.colorSpace;
    swapchainCreateInfo.imageArrayLayers = 1; // We're not rendering in stereoscopic 3D => set this to 1
    swapchainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // Use eColorAttachment so we can directly render to the swapchain images.

    auto queueFamilyVector = std::vector<uint32_t>({context.queueFamilies.graphicsFamily, context.queueFamilies.presentFamily});
    if (context.queueFamilies.graphicsFamily != context.queueFamilies.presentFamily) {
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

    FATAL_ERROR_UNLESS(swapchainRenderPass.colorAttachmentFormat == context.surfaceFormat.format, "Render pass using for render to swapchain doesn't render to correct format.")

    // Make framebuffers
    {
        framebuffers.clear();
        for (const auto& image : images) {
            framebuffers.emplace_back(context, image, context.surfaceFormat.format, context.windowSize, *swapchainRenderPass);
        }
    }
}
