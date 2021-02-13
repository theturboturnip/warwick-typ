//
// Created by samuel on 11/02/2021.
//

#include <imgui_impl_vulkan.h>
#include "VulkanSimAppData.h"

VulkanSimAppData::VulkanSimAppData(VulkanSimAppData::Global&& globalData,
                                   std::vector<VulkanSimFrameData>& bufferList,
                                   VulkanSwapchain& swapchain) : globalData(globalData) {
    FATAL_ERROR_IF(bufferList.empty(), "Empty list of buffers");
    frameData.reserve(bufferList.size());
    for (uint32_t i = 0; i < bufferList.size(); i++) {
        frameData.emplace_back(this->globalData, this->globalData.context, i, &bufferList[i]);
    }

    swapchainImageData.reserve(swapchain.imageCount);
    for (uint32_t i = 0; i < swapchain.imageCount; i++) {
        swapchainImageData.emplace_back(i, &swapchain.framebuffers[i]);
    }
}

VulkanSimAppData::PerFrameData::PerFrameData(VulkanSimAppData::Global& globalData, VulkanContext& context, uint32_t index, VulkanSimFrameData *buffers)
    : index(index),
      buffers(buffers),

      simBuffersImage(
          context,
          vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
          {
              globalData.simSize.padded_pixel_size.x * 2,
              globalData.simSize.padded_pixel_size.y * 2,
          },
          vk::Format::eR32G32B32A32Sfloat,
          true
      ),
      simBuffersSampler(context, simBuffersImage),
      simBuffersImageDescriptorSet(
          globalData.pipelines.buildFullscreenPressureDescriptors(context, simBuffersSampler)
      ),
      simBuffersDescriptorSet(
          globalData.pipelines.buildComputeSimDataImageDescriptors(globalData.context, *buffers, *simBuffersImage, simBuffersSampler)
      ),

      vizFramebuffer(
          context,
          vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
          globalData.simSize.padded_pixel_size,
          globalData.vizRenderPass
      ),
      vizFramebufferDescriptorSet(
          ImGui_ImplVulkan_MakeDescriptorSet(vizFramebuffer.getImageView()),
          vk::PoolFree(*context.device, *context.descriptorPool, VULKAN_HPP_DEFAULT_DISPATCHER)
      ),

      imageAcquired(*context.device),
      simFinished(*context.device),
      renderFinishedShouldPresent(*context.device),
      renderFinishedShouldSim(*context.device),
      computeFinished(*context.device),
      inFlight(context, true),

      threadOutputs({
          .commandBuffer = std::move(context.allocateCommandBuffers(vk::CommandBufferLevel::ePrimary, 1)[0])
      })
    {}

VulkanSimAppData::PerSwapchainImageData::PerSwapchainImageData(uint32_t index, VulkanFramebuffer* framebuffer)
    : index(index),
        framebuffer(framebuffer),
        inFlight(nullptr)
{}
