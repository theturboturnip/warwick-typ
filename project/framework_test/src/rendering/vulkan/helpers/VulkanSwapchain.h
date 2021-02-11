//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "VulkanRenderPass.h"
#include "rendering/vulkan/VulkanContext.h"
#include "VulkanFramebuffer.h"

#include <vector>
#include <functional>
#include <vulkan/vulkan.hpp>

class VulkanSwapchain {
public:
    vk::UniqueSwapchainKHR swapchain;
    vk::Extent2D extents;
    uint32_t imageCount;
    std::vector<vk::Image> images;
    std::vector<VulkanFramebuffer> framebuffers;

    VulkanSwapchain(VulkanContext& setup, VulkanRenderPass& swapchainRenderPass);
    VulkanSwapchain(VulkanSwapchain&&) noexcept = default;

    const vk::SwapchainKHR& operator *() const{
        return *swapchain;
    }
};
