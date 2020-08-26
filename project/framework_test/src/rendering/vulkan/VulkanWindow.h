//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <SDL.h>
#include <vulkan/vulkan.hpp>

#include "VulkanPipelineSet.h"
#include "VulkanQueueFamilies.h"
#include "VulkanRenderPass.h"
#include "VulkanShader.h"
#include "util/Size.h"

extern VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

class VulkanWindow {
    Size<size_t> window_size;

    SDL_Window* window;
    vk::UniqueInstance instance;
    vk::DispatchLoaderDynamic dispatch_loader;
    vk::UniqueSurfaceKHR surface;
    // This uses a dynamic loader, becuase the loader functions vkCreateDebugUtilsMessengerEXT etc. need to be dynamically linked at runtime
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> debug_messenger;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueDevice logicalDevice;
    // TODO - this isn't necessary to keep in the class
    VulkanQueueFamilies queueFamilies;
    vk::Queue graphicsQueue, presentQueue;
    vk::UniqueCommandPool cmdPool;
    vk::UniqueDescriptorPool descriptorPool;

    struct {
        vk::SurfaceFormatKHR surfaceFormat;
        vk::PresentModeKHR presentMode;
        vk::Extent2D extents;
        uint32_t imageCount;
    } swapchainProps;
    vk::UniqueSwapchainKHR swapchain;
//    struct SWFrameData {
//        vk::Image image;
//        vk::UniqueImageView imageView;
//        vk::UniqueFramebuffer framebuffer;
//        vk::UniqueCommandBuffer cmdBuffer;
//    };
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueImageView> swapchainImageViews;
    std::vector<vk::UniqueFramebuffer> swapchainFramebuffers;
    std::vector<vk::UniqueCommandBuffer> perFrameCommandBuffers;
    //vk::UniqueCommandBuffer imguiCmdBuffer;

    VulkanRenderPass renderPass;

    std::unique_ptr<VulkanPipelineSet> pipelines;

    // The tutorial creates multiple sets of semaphores so that multiple frames can be in-flight at once.
    // i.e. multiple frames can be drawn at once.
    // However, we aren't planning on drawing multiple frames at once. The GPU will be busy most of the time doing CUDA work.
    // So we only create two semaphores - has image, and finished rendering.
    vk::UniqueSemaphore hasImage, renderFinished;

    friend class SystemThreadWorker;
public:
    VulkanWindow(const vk::ApplicationInfo& info, Size<size_t> window_size);
    VulkanWindow(const VulkanWindow&) = delete;
    VulkanWindow(VulkanWindow&&) = delete; // TODO - I think this is right, we shouldn't move stuff because it may depend on pointers???
    ~VulkanWindow();

    // TODO - this will change in the future
    void main_loop();

private:
    void check_sdl_error(SDL_bool success);
    void check_vulkan_error(vk::Result result);

    vk::UniqueImageView make_identity_view(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectFlags = vk::ImageAspectFlagBits::eColor);

    // TODO - unused, remove
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