//
// Created by samuel on 08/02/2021.
//

#pragma once

#include <SDL.h>
#include <vulkan/vulkan.hpp>

#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "rendering/vulkan/helpers/VulkanSwapchain.h"
#include "util/Size.h"

/*
 * Creates SDL Window, vk instance, vk physical device, vk logical device.
 */
class VulkanSetup {
public:
    vk::UniqueInstance instance;

    Size<uint32_t> windowSize;
    SDL_Window* window;
    vk::UniqueSurfaceKHR surface;

    vk::DispatchLoaderDynamic dynamicLoader;
    // This uses a dynamic loader, becuase the loader functions vkCreateDebugUtilsMessengerEXT etc. need to be dynamically linked at runtime
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugMessenger;

    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;

    VulkanQueueFamilies queueFamilies;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    VulkanSetup(vk::ApplicationInfo appInfo, Size<uint32_t> windowSize);
    VulkanSetup(VulkanSetup&&) = delete;
    ~VulkanSetup();
};
