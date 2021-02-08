//
// Created by samuel on 08/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>

class VulkanSwapchain {
public:
    vk::UniqueSwapchainKHR swapchain;
    vk::SurfaceFormatKHR format;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extents;
    std::vector<vk::Image> images;
    std::vector<vk::UniqueImageView> imageViews;
    std::vector<vk::UniqueFramebuffer> framebuffers;

    VulkanSwapchain();
    VulkanSwapchain(VulkanSwapchain&&) noexcept = default;

    const vk::SwapchainKHR& operator *() const{
        return *swapchain;
    }
};
