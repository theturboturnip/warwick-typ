//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <SDL.h>
#include <simulation/SimulationBackendEnum.h>
#include <simulation/file_format/FluidParams.h>
#include <simulation/file_format/SimSnapshot.h>
#include <vulkan/vulkan.hpp>

#include "VulkanSetup.h"
#include "VulkanSimPipelineSet.h"
#include "VulkanSimSemaphoreSet.h"
#include "rendering/vulkan/helpers/VulkanFence.h"
#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "rendering/vulkan/helpers/VulkanRenderPass.h"
#include "rendering/vulkan/helpers/VulkanSemaphore.h"
#include "rendering/vulkan/helpers/VulkanShader.h"
#include "util/Size.h"

extern VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebug(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

class VulkanSimApp {
    Size<uint32_t> windowSize;
    VulkanSetup setup;
    vk::Device device;

    vk::UniqueCommandPool cmdPool;
    vk::UniqueDescriptorPool descriptorPool;

    struct {
        vk::SurfaceFormatKHR surfaceFormat;
        vk::PresentModeKHR presentMode;
        vk::Extent2D extents;
        uint32_t imageCount;
    } swapchainProps;
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueImageView> swapchainImageViews;
    std::vector<vk::UniqueFramebuffer> swapchainFramebuffers;
    std::vector<vk::UniqueCommandBuffer> perFrameCommandBuffers;

    VulkanRenderPass imguiRenderPass;
    VulkanRenderPass simRenderPass;

    std::unique_ptr<VulkanSimPipelineSet> pipelines;

    // The tutorial creates multiple sets of semaphores so that multiple frames can be in-flight at once.
    // i.e. multiple frames can be drawn at once.
    // However, we aren't planning on drawing multiple frames at once. The GPU will be busy most of the time doing CUDA work.
    // So we only create two semaphores - has image, and finished rendering.
    std::unique_ptr<VulkanSimSemaphoreSet> semaphores;
    std::unique_ptr<VulkanFence> graphicsFence;

    friend class SystemWorker;
public:
    VulkanSimApp(const vk::ApplicationInfo& appInfo, Size<uint32_t> windowSize);
    VulkanSimApp(const VulkanSimApp &) = delete;
    VulkanSimApp(VulkanSimApp &&) = delete;

    // TODO - this will change in the future
    void main_loop(SimulationBackendEnum backendType, const FluidParams &params, const SimSnapshot &snapshot);

#if CUDA_ENABLED
    SimSnapshot test_cuda_sim(const FluidParams& params, const SimSnapshot& snapshot);
#endif

private:
    vk::UniqueImageView make_identity_view(vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectFlags = vk::ImageAspectFlagBits::eColor) const;
};