//
// Created by samuel on 11/02/2021.
//

#pragma once

#include <vector>
#include <imgui.h>
#include <simulation/file_format/SimSize.h>
#include <rendering/vulkan/helpers/VulkanBackedFramebuffer.h>
#include <rendering/vulkan/helpers/VulkanSemaphore.h>
#include <rendering/vulkan/helpers/VulkanFence.h>
#include "VulkanContext.h"
#include "VulkanSimPipelineSet.h"

class VulkanSimAppData {
public:
    struct Global {
        VulkanContext& context;
        ImGuiContext* imguiContext;

        SimSize simSize;

        vk::RenderPass imguiRenderPass;
        vk::Rect2D imguiRenderArea;

        vk::RenderPass simRenderPass;
        vk::Rect2D simRenderArea;

        VulkanSimPipelineSet& pipelines;
    };

    class PerFrameData {
    public:
        uint32_t index;

        // Simulation buffer related data.
        VulkanSimFrameData* buffers;
        vk::UniqueDescriptorSet simBuffersDescriptorSet;

        // The framebuffer to render the visualization into, and a descriptor set that samples it for ImGui.
        VulkanBackedFramebuffer vizFramebuffer;
        vk::UniqueDescriptorSet vizFramebufferDescriptorSet;

        // Synchronization
        VulkanSemaphore imageAcquired, simFinished, renderFinishedShouldPresent, renderFinishedShouldSim;
        VulkanFence inFlight; // Used to wait for the render to finish

        // Data which should only be accessed by the worker thread.
        struct {
            vk::UniqueCommandBuffer commandBuffer;
        } threadOutputs;

        PerFrameData(Global& globalData, VulkanContext& context, uint32_t index, VulkanSimFrameData* buffers);
        PerFrameData(PerFrameData&&) noexcept = default;
    };

    // The swapchain could have more or fewer images than we have PerFrameDatas,
    // so we track them separately from PerFrameData.
    class PerSwapchainImageData {
    public:
        uint32_t index;

        VulkanFramebuffer* framebuffer;
        // Reference to the VulkanFence of the Sim Frame that is currently rendering to this.
        vk::Fence inFlight;

        PerSwapchainImageData(VulkanContext& context, uint32_t index, VulkanFramebuffer* framebuffer);
        PerSwapchainImageData(PerSwapchainImageData&&) noexcept = default;
    };

    Global globalData;
    std::vector<PerFrameData> frameData;
    std::vector<PerSwapchainImageData> swapchainImageData;

    VulkanSimAppData(Global&& globalData, std::vector<VulkanSimFrameData>& bufferList);
    VulkanSimAppData(VulkanSimAppData&&) noexcept = default;
    VulkanSimAppData(const VulkanSimAppData&) = delete;
};