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
#include "rendering/vulkan/helpers/VulkanFence.h"
#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "rendering/vulkan/helpers/VulkanRenderPass.h"
#include "rendering/vulkan/helpers/VulkanSemaphore.h"
#include "rendering/vulkan/helpers/VulkanShader.h"
#include "util/Size.h"
#include "imgui.h"

class VulkanSimApp {
    VulkanContext context;
    vk::Device device;

    VulkanRenderPass finalCompositeRenderPass;
    VulkanRenderPass vizRenderPass;

    VulkanSwapchain swapchain;

    // TODO - move into VulkanContext
    ImGuiContext* imContext;

    friend class SystemWorker;
public:
    VulkanSimApp(const vk::ApplicationInfo& appInfo, Size<uint32_t> windowSize);
    VulkanSimApp(const VulkanSimApp &) = delete;
    VulkanSimApp(VulkanSimApp &&) = delete;
    ~VulkanSimApp();

    // TODO - this will change in the future
    void main_loop(SimulationBackendEnum backendType, const FluidParams &params, const SimSnapshot &snapshot);

#if CUDA_ENABLED
    SimSnapshot test_cuda_sim(const FluidParams& params, const SimSnapshot& snapshot);
#endif
};