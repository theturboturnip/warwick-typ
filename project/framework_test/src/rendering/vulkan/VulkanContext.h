//
// Created by samuel on 08/02/2021.
//

#pragma once

#include <SDL.h>
#include <vulkan/vulkan.hpp>

#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "util/Size.h"

/*
 * Creates SDL Window, vk instance, vk physical device, vk logical device.
 */
class VulkanContext {
public:
    vk::UniqueInstance instance;

    Size<uint32_t> windowSize;
    SDL_Window* window;
    vk::UniqueSurfaceKHR surface;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;

    vk::DispatchLoaderDynamic dynamicLoader;
    // This uses a dynamic loader, becuase the loader functions vkCreateDebugUtilsMessengerEXT etc. need to be dynamically linked at runtime
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debugMessenger;

    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice device;

    VulkanQueueFamilies queueFamilies;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::Queue computeQueue;

    vk::UniqueCommandPool graphicsCmdPool;
    vk::UniqueCommandPool computeCmdPool;
    vk::UniqueDescriptorPool descriptorPool;

    VulkanContext(vk::ApplicationInfo appInfo, Size<uint32_t> windowSize);
    VulkanContext(VulkanContext&&) = delete;
    ~VulkanContext();

    uint32_t selectMemoryTypeIndex(vk::MemoryRequirements requirements, vk::MemoryPropertyFlags properties) const;
    std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(vk::UniqueCommandPool& pool, vk::CommandBufferLevel level, size_t count);
};
