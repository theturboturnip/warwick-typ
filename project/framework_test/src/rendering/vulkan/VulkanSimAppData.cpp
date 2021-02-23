//
// Created by samuel on 11/02/2021.
//

#include <imgui_impl_vulkan.h>
#include "VulkanSimAppData.h"
#include "rendering/shaders/global_structures.h"

VulkanSimAppData::VulkanSimAppData(VulkanSimAppData::Global&& globalData,
                                   std::vector<VulkanSimFrameData>& bufferList,
                                   VulkanSwapchain& swapchain) : globalData(globalData), sharedFrameData(globalData, globalData.context) {
    FATAL_ERROR_IF(bufferList.empty(), "Empty list of buffers");
    perFrameData.reserve(bufferList.size());
    for (uint32_t i = 0; i < bufferList.size(); i++) {
        perFrameData.emplace_back(this->globalData, this->globalData.context, i, &bufferList[i]);
    }

    swapchainImageData.reserve(swapchain.imageCount);
    for (uint32_t i = 0; i < swapchain.imageCount; i++) {
        swapchainImageData.emplace_back(i, &swapchain.framebuffers[i]);
    }
}

VulkanSimAppData::PerFrameData::PerFrameData(VulkanSimAppData::Global& globalData, VulkanContext& context, uint32_t index, VulkanSimFrameData *buffers)
    : index(index),
      buffers(buffers),

      simBufferCopyInput_comp_ds(
              globalData.pipelines.buildSimBufferCopyInput_comp_ds(globalData.context, *buffers)
      ),

      simFinishedCanBufferCopy(*context.device),
      bufferCopyFinishedCanSim(*context.device),
      bufferCopyFinishedCanCompute(*context.device),
      computeFinishedCanRender(*context.device),
      imageAcquiredCanRender(*context.device),
      renderFinishedNextFrameCanBufferCopy(*context.device),
      renderFinishedCanPresent(*context.device),
      frameCmdBuffersInUse(context, true),

      bufferCopyCmdBuffer(std::move(context.allocateCommandBuffers(context.computeCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0])),
      computeCmdBuffer(std::move(context.allocateCommandBuffers(context.computeCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0])),
      renderCmdBuffer(std::move(context.allocateCommandBuffers(context.computeCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0]))
    {}

VulkanSimAppData::PerSwapchainImageData::PerSwapchainImageData(uint32_t index, VulkanFramebuffer* framebuffer)
    : index(index),
        framebuffer(framebuffer)//,
//        inFlight(nullptr)
{}

VulkanSimAppData::SharedFrameData::SharedFrameData(VulkanSimAppData::Global &globalData, VulkanContext &context)
    :  simDataImage(
                context,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                {
                        globalData.simSize.padded_pixel_size.x * 2,
                        globalData.simSize.padded_pixel_size.y * 2,
                },
                vk::Format::eR32G32B32A32Sfloat,
                true
        ),
        simDataSampler(context, simDataImage),

        simDataSampler_comp_ds(
                globalData.pipelines.buildSimDataSampler_comp_ds(context, simDataSampler)
        ),
        simDataSampler_frag_ds(
                globalData.pipelines.buildSimDataSampler_frag_ds(context, simDataSampler)
        ),

        simBufferCopyOutput_comp_ds(
            globalData.pipelines.buildSimBufferCopyOutput_comp_ds(globalData.context, simDataSampler)
        ),

        particleBuffer(context,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
            globalData.props.maxParticles * sizeof(Shaders::Particle),
            true // Shared between graphics and compute
        ),
        particleInputBuffer_comp_ds(
               globalData.pipelines.buildParticleInputBuffer_comp_ds(
                       context, particleBuffer.asDescriptor()
               )
        ),
        particleInputBuffer_vert_ds(
               globalData.pipelines.buildParticleInputBuffer_vert_ds(
                       context, particleBuffer.asDescriptor()
               )
        ),
        particleOutputBuffer_comp_ds(
               globalData.pipelines.buildParticleOutputBuffer_comp_ds(
                       context, particleBuffer.asDescriptor()
               )
        ),

        vizFramebuffer(
               context,
               vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
               {
                       globalData.simSize.padded_pixel_size.x * 2,
                       globalData.simSize.padded_pixel_size.y * 2,
               },
               globalData.vizRenderPass
       ),
       vizFramebufferDescriptorSet(
               ImGui_ImplVulkan_MakeDescriptorSet(vizFramebuffer.getImageView()),
               // TODO - technically this is the incorrect pool
               vk::PoolFree(*context.device, *context.descriptorPool, VULKAN_HPP_DEFAULT_DISPATCHER)
       )
{}
