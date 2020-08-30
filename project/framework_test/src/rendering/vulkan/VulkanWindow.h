//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <SDL.h>
#include <simulation/SimulationBackendEnum.h>
#include <simulation/file_format/FluidParams.h>
#include <simulation/file_format/SimSnapshot.h>
#include <vulkan/vulkan.hpp>

#include "VulkanPipelineSet.h"
#include "VulkanQueueFamilies.h"
#include "VulkanRenderPass.h"
#include "VulkanSemaphore.h"
#include "VulkanSemaphoreSet.h"
#include "VulkanShader.h"
#include "util/Size.h"

extern VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

class VulkanWindow {
    Size<uint32_t> window_size;

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

    VulkanRenderPass imguiRenderPass;
    VulkanRenderPass simRenderPass;

    std::unique_ptr<VulkanPipelineSet> pipelines;

    // The tutorial creates multiple sets of semaphores so that multiple frames can be in-flight at once.
    // i.e. multiple frames can be drawn at once.
    // However, we aren't planning on drawing multiple frames at once. The GPU will be busy most of the time doing CUDA work.
    // So we only create two semaphores - has image, and finished rendering.
    std::unique_ptr<VulkanSemaphoreSet> semaphores;

    friend class SystemWorker;
public:
    VulkanWindow(const vk::ApplicationInfo& info, Size<uint32_t> window_size);
    VulkanWindow(const VulkanWindow&) = delete;
    VulkanWindow(VulkanWindow&&) = delete;
    ~VulkanWindow();

    // TODO - this will change in the future
    void main_loop(SimulationBackendEnum backendType, const FluidParams &params, const SimSnapshot &snapshot);

#if CUDA_ENABLED
    SimSnapshot test_cuda_sim(const FluidParams& params, const SimSnapshot& snapshot);
#endif

private:
    void check_sdl_error(SDL_bool success) const;
    void check_vulkan_error(vk::Result result) const;

    vk::UniqueImageView make_identity_view(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectFlags = vk::ImageAspectFlagBits::eColor) const;

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