//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <SDL.h>
#include <vulkan/vulkan.hpp>

#include "VulkanQueueFamilies.h"
#include "util/Size.h"

#if !NDEBUG
extern VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
#endif

class VulkanWindow {
    SDL_Window* window;
    vk::UniqueInstance instance;
    vk::DispatchLoaderDynamic dispatch_loader;
    vk::UniqueSurfaceKHR surface;
    // This uses a dynamic loader, becuase the loader functions vkCreateDebugUtilsMessengerEXT etc. need to be dynamically linked at runtime
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debug_messenger;
    vk::PhysicalDevice physicalDevice;
    VulkanQueueFamilies queueFamilies;
public:
    VulkanWindow(const vk::ApplicationInfo& info, Size<size_t> window_size);
    ~VulkanWindow();

    // TODO - this will change in the future
    void main_loop();

    void check_sdl_error(SDL_bool success);
    void check_vulkan_error(vk::Result result);

    template<typename FuncPtrType>
    FuncPtrType get_vulkan_function(const char* name) {
        static_assert(std::is_pointer<FuncPtrType>::value && std::is_function<std::remove_pointer_t<FuncPtrType>>::value,
                      "FuncPtrType must be a pointer to a function.");
        return (FuncPtrType)vkGetInstanceProcAddr(*instance, name);
    }
#if !NDEBUG
    constexpr static bool VulkanDebug = true;
#else
    constexpr static bool VulkanDebug = false;
#endif
};