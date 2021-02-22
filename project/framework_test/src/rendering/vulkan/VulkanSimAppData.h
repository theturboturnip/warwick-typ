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
#include <rendering/vulkan/helpers/VulkanSwapchain.h>
#include <rendering/vulkan/helpers/VulkanBackedBuffer.h>
#include <rendering/vulkan/helpers/VulkanBackedGPUBuffer_WithStaging.h>
#include "VulkanContext.h"
#include "VulkanSimPipelineSet.h"

class VulkanSimAppData {
public:
    struct Global {
        VulkanContext& context;
        ImGuiContext* imguiContext;
        SimAppProperties props;

        SimSize simSize;

        vk::RenderPass finalCompositeRenderPass;
        vk::Rect2D finalCompositeRect;

        vk::RenderPass vizRenderPass;
        vk::Rect2D vizRect;

        VulkanSimPipelineSet& pipelines;
    };

    class PerFrameData {
    public:
        uint32_t index;

        // Simulation buffer related data.
        VulkanSimFrameData* buffers;
        // Image generated by a compute shader storing all simulation buffers.
        // .x = u
        // .y = v
        // .z = p
        // .w = fluidmask
        VulkanBackedGPUImage simDataImage;
        VulkanImageSampler simDataSampler;
        // SimBuffers Image descriptor set containing simBuffersImage, used in fragment shader
        vk::UniqueDescriptorSet simDataSampler_comp_ds;
        vk::UniqueDescriptorSet simDataSampler_frag_ds;
        // SimBuffers descriptor set containing the buffers + a writable simBuffersImage, used in compute shader
        vk::UniqueDescriptorSet simBuffers_comp_ds;
        // TODO comment these lol
        VulkanBackedGPUBuffer_WithStaging particleBuffer;
        vk::UniqueDescriptorSet particleInputBuffer_comp_ds;
        vk::UniqueDescriptorSet particleInputBuffer_vert_ds;
        vk::UniqueDescriptorSet particleOutputBuffer_comp_ds;


        // The framebuffer to render the visualization into, and a descriptor set that samples it for ImGui.
        VulkanBackedFramebuffer vizFramebuffer;
        vk::UniqueDescriptorSet vizFramebufferDescriptorSet;

        // Synchronization
        VulkanSemaphore imageAcquired, simFinished, renderFinishedShouldPresent, renderFinishedShouldCompute, computeFinishedShouldSim, computeFinished;
        VulkanFence inFlight; // Used to wait for the render to finish

        // Data which should only be accessed by the worker thread.
        struct {
            vk::UniqueCommandBuffer computeCommandBuffer;
            vk::UniqueCommandBuffer graphicsCommandBuffer;
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

        PerSwapchainImageData(uint32_t index, VulkanFramebuffer* framebuffer);
        PerSwapchainImageData(PerSwapchainImageData&&) noexcept = default;
    };

    Global globalData;
    std::vector<PerFrameData> frameData;
    std::vector<PerSwapchainImageData> swapchainImageData;

    VulkanSimAppData(Global&& globalData, std::vector<VulkanSimFrameData>& bufferList, VulkanSwapchain& swapchain);
    VulkanSimAppData(VulkanSimAppData&&) noexcept = default;
    VulkanSimAppData(const VulkanSimAppData&) = delete;
};