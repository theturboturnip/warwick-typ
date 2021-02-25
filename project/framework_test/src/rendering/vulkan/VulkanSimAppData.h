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

        vk::Device device;
        VulkanSwapchain& swapchain;

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
        // SimBuffers descriptor set containing the buffers, used in compute shader
        vk::UniqueDescriptorSet simBufferCopyInput_comp_ds;

        // ParticleEmitters[props.maxParticleEmitters] array
        VulkanBackedGPUBuffer_WithStaging particleEmitters;
        vk::UniqueDescriptorSet particleEmitters_comp_ds;

        // Synchronization

        // Signalled when the Sim relinquishes this frame's buffers to the Compute phase.
        VulkanSemaphore simFinishedCanCompute;
        // Signalled when the Compute phase is finished, and relinquishes the sim buffers back to the Sim.
        // The Render phase could now use the computed data to render.
        VulkanSemaphore computeFinishedCanSim, computeFinishedCanRender;
        // Signalled when the specified swapchain image is ready for the Render phase to start rendering.
        // TODO - This prevents even the normal visualization from rendering until the image is acquired.
        //  in theory this could be slower than we want.
        //  we could fix this by separating Render into VizRender and FinalRender, where FinalRender renders directly to the swapchain.
        //  but this is too much effort for now.
        VulkanSemaphore imageAcquiredCanRender;
        // Signalled when the Render phase finishes.
        // The next frame can start their BufferCopy phase, and we can Present the new image.
        VulkanSemaphore renderFinishedNextFrameCanCompute, renderFinishedCanPresent;

        // Closed while the command buffers are being recorded and executing.
        // Used to stop us from trying to re-record data while this is happening.
        VulkanFence frameCmdBuffersInUse;

        vk::UniqueCommandBuffer computeCmdBuffer;
        vk::UniqueCommandBuffer renderCmdBuffer;

        PerFrameData(Global& globalData, VulkanContext& context, uint32_t index, VulkanSimFrameData* buffers);
        PerFrameData(PerFrameData&&) noexcept = default;
    };

    class SharedFrameData {
    public:
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
        // SimBuffers descriptor set containing a writable simBuffersImage, used in compute shader
        vk::UniqueDescriptorSet simBufferCopyOutput_comp_ds;

        // ParticleEmitData[props.maxParticlesEmittedPerFrame] array
        // tells the compute_particle_emit shader where to put emitted particles
        VulkanBackedBuffer particlesToEmit;
        vk::UniqueDescriptorSet particlesToEmit_comp_ds;
        // Particle[props.maxParticles] array
        VulkanBackedBuffer particleDataArray;
        vk::UniqueDescriptorSet particleDataArray_comp_ds;
        vk::UniqueDescriptorSet particleDataArray_vert_ds;
        // Growable/Shrinkable list of particle indices into particleDataArray, listing inactive particles.
        VulkanBackedGPUBuffer_WithStaging inactiveParticleIndexList;
        vk::UniqueDescriptorSet inactiveParticleIndexList_comp_ds;
        // Data to reset the inactiveParticleIndexList with if necessary
        VulkanBackedGPUBuffer_WithStaging inactiveParticleIndexList_resetData;
        // Growable list of particle indices, with an extra atomic uint32_t size.
        VulkanBackedBuffer particleIndexSimulateList, particleIndexDrawList;
        vk::UniqueDescriptorSet particleIndexSimulateList_comp_ds;
        vk::UniqueDescriptorSet particleIndexDrawList_comp_ds;
        vk::UniqueDescriptorSet particleIndexDrawList_vert_ds;
        // ParticleIndirectCommands
        VulkanBackedBuffer particleIndirectCommands;
        vk::UniqueDescriptorSet particleIndirectCommands_comp_ds;
        // Vertex[] of particle data (triangle strip)
        VulkanBackedGPUBuffer_WithStaging particleVertexData;

        // The framebuffer to render the visualization into, and a descriptor set that samples it for ImGui.
        VulkanBackedFramebuffer vizFramebuffer;
        vk::UniqueDescriptorSet vizFramebufferDescriptorSet;

        SharedFrameData(Global& globalData, VulkanContext& context);
        SharedFrameData(SharedFrameData&&) noexcept = default;
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
    SharedFrameData sharedFrameData;
    std::vector<PerFrameData> perFrameData;
    std::vector<PerSwapchainImageData> swapchainImageData;

    VulkanSimAppData(Global&& globalData, std::vector<VulkanSimFrameData>& bufferList, VulkanSwapchain& swapchain);
    VulkanSimAppData(VulkanSimAppData&&) noexcept = default;
    VulkanSimAppData(const VulkanSimAppData&) = delete;
};