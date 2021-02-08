//
// Created by samuel on 22/08/2020.
//

#pragma once

#include <SDL.h>
#include <rendering/vulkan/helpers/VulkanSwapchain.h>
#include <simulation/SimulationBackendEnum.h>
#include <simulation/file_format/FluidParams.h>
#include <simulation/file_format/SimSnapshot.h>
#include <vulkan/vulkan.hpp>

#include "VulkanContext.h"
#include "VulkanSimPipelineSet.h"
#include "VulkanSimSemaphoreSet.h"
#include "rendering/vulkan/helpers/VulkanFence.h"
#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "rendering/vulkan/helpers/VulkanRenderPass.h"
#include "rendering/vulkan/helpers/VulkanSemaphore.h"
#include "rendering/vulkan/helpers/VulkanShader.h"
#include "util/Size.h"

class VulkanSimApp {
    VulkanContext context;
    vk::Device device;

    VulkanRenderPass imguiRenderPass;
    VulkanRenderPass simRenderPass;

    VulkanSwapchain swapchain;

    vk::UniqueCommandPool cmdPool;
    vk::UniqueDescriptorPool descriptorPool;

    std::vector<vk::UniqueCommandBuffer> perFrameCommandBuffers;

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
};