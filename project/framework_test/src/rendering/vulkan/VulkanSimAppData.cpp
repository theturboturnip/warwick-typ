//
// Created by samuel on 11/02/2021.
//

#include <imgui_impl_vulkan.h>
#include <rendering/vulkan/viz/vertex.h>
#include "VulkanSimAppData.h"
#include "rendering/shaders/global_structures.h"

vk::UniqueDescriptorSet bufferDescSet_comp(VulkanSimAppData::Global &globalData, vk::DescriptorBufferInfo info) {
    return globalData.pipelines.buildBuffer_comp_ds(globalData.context, info);
}
vk::UniqueDescriptorSet bufferDescSet_vert(VulkanSimAppData::Global &globalData, vk::DescriptorBufferInfo info) {
    return globalData.pipelines.buildBuffer_vert_ds(globalData.context, info);
}
vk::UniqueDescriptorSet bufferDescSet_frag(VulkanSimAppData::Global &globalData, vk::DescriptorBufferInfo info) {
    return globalData.pipelines.buildBuffer_frag_ds(globalData.context, info);
}


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

      particleEmitters(context, vk::BufferUsageFlagBits::eStorageBuffer, globalData.props.maxParicleEmitters * sizeof(Shaders::ParticleEmitter)),
      particleEmitters_comp_ds(bufferDescSet_comp(globalData, particleEmitters.getGpuDescriptor())),

      quantityScalar_range(context, vk::BufferUsageFlagBits::eStorageBuffer, sizeof(Shaders::FloatRange)),
      quantityScalar_range_frag_ds(bufferDescSet_frag(globalData, quantityScalar_range.getGpuDescriptor())),

      simFinishedCanCompute(*context.device),
      computeFinishedCanSim(*context.device),
      computeFinishedCanRender(*context.device),
      imageAcquiredCanRender(*context.device),
      renderFinishedNextFrameCanCompute(*context.device),
      renderFinishedCanPresent(*context.device),
      frameCmdBuffersInUse(context, true),

      computeCmdBuffer(std::move(context.allocateCommandBuffers(context.computeCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0])),
      renderCmdBuffer(std::move(context.allocateCommandBuffers(context.graphicsCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0]))
    {}

VulkanSimAppData::PerSwapchainImageData::PerSwapchainImageData(uint32_t index, VulkanFramebuffer* framebuffer)
    : index(index),
        framebuffer(framebuffer),
        inFlight(nullptr)
{}

VulkanSimAppData::SharedFrameData::SharedFrameData(VulkanSimAppData::Global &globalData, VulkanContext &context)
    :  simDataImage(
                context,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                {
                    // Subtract 1 because at the bottom/right ends we don't want double boundaries
                    globalData.simSize.internal_pixel_size.x * 2 - 1,
                    globalData.simSize.internal_pixel_size.y * 2 - 1,
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

        particlesToEmit(
            context,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            globalData.props.maxParticlesEmittedPerFrame * sizeof(Shaders::ParticleToEmitData),
            true
        ),
        particlesToEmit_comp_ds(bufferDescSet_comp(globalData, particlesToEmit.asDescriptor())),
        particleDataArray(context,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            globalData.props.maxParticles * sizeof(Shaders::Particle),
            true // Shared between graphics and compute
        ),
        particleDataArray_comp_ds(bufferDescSet_comp(globalData, particleDataArray.asDescriptor())),
        particleDataArray_vert_ds(bufferDescSet_vert(globalData, particleDataArray.asDescriptor())),
        inactiveParticleIndexList(
            context,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            (1 + globalData.props.maxParticles) * sizeof(uint32_t),
            false
        ),
        inactiveParticleIndexList_comp_ds(bufferDescSet_comp(globalData, inactiveParticleIndexList.getGpuDescriptor())),
        inactiveParticleIndexList_resetData(
               context,
               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
               (1 + globalData.props.maxParticles) * sizeof(uint32_t),
               false
        ),
        particleIndexSimulateList(
            context,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            (1 + globalData.props.maxParticles) * sizeof(uint32_t),
            false // not shared
        ),
        particleIndexDrawList(
           context,
           vk::MemoryPropertyFlagBits::eDeviceLocal,
           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
           (1 + globalData.props.maxParticles) * sizeof(uint32_t),
           false // not shared
        ),
        particleIndexSimulateList_comp_ds(bufferDescSet_comp(globalData, particleIndexSimulateList.asDescriptor())),
        particleIndexDrawList_comp_ds(bufferDescSet_comp(globalData, particleIndexDrawList.asDescriptor())),
        particleIndexDrawList_vert_ds(bufferDescSet_vert(globalData, particleIndexDrawList.asDescriptor())),
        particleIndirectCommands(
           context,
           vk::MemoryPropertyFlagBits::eDeviceLocal,
           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
           sizeof(Shaders::ParticleIndirectCommands),
           true // shared
        ),
        particleIndirectCommands_comp_ds(bufferDescSet_comp(globalData, particleIndirectCommands.asDescriptor())),
        particleVertexData(
           context,
           vk::BufferUsageFlagBits::eVertexBuffer,
           sizeof(Vertex) * 4,
           false // not shared
        ),
//        particleInputBuffer_comp_ds(
//               globalData.pipelines.buildParticleInputBuffer_comp_ds(
//                       context, particleBuffer.asDescriptor()
//               )
//        ),
//        particleInputBuffer_vert_ds(
//               globalData.pipelines.buildParticleInputBuffer_vert_ds(
//                       context, particleBuffer.asDescriptor()
//               )
//        ),
//        particleOutputBuffer_comp_ds(
//               globalData.pipelines.buildParticleOutputBuffer_comp_ds(
//                       context, particleBuffer.asDescriptor()
//               )
//        ),

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
{
    // Alloc a command buffer to init data (auto deleted)
    vk::UniqueCommandBuffer buf = std::move(globalData.context.allocateCommandBuffers(globalData.context.computeCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0]);

    const auto beginInfo = vk::CommandBufferBeginInfo{};
    buf->begin(&beginInfo);

    // INITIALIZE VERTEX BUFFER
    {
        auto memory = particleVertexData.mapCPUMemory(*globalData.context.device);
        auto* vData = (Vertex*)(*memory);
        vData[0] = Vertex{
            .pos = glm::vec2(0, -1),
            .uv = glm::vec2(0, 0),
        };
        vData[1] = Vertex{
                .pos = glm::vec2(-1, 0),
                .uv = glm::vec2(0, 1),
        };
        vData[2] = Vertex{
                .pos = glm::vec2(1, 0),
                .uv = glm::vec2(1, 0),
        };
        vData[3] = Vertex{
                .pos = glm::vec2(0, 1),
                .uv = glm::vec2(1, 1),
        };

        // Auto unmapped
    }
    particleVertexData.scheduleCopyToGPU(*buf);

    // INITIALIZE INACTIVE LIST WITH ALL PARTICLES
    {
        const auto maxParticles = globalData.props.maxParticles;
        auto inactiveNumbers = std::vector<uint32_t>(1 + maxParticles);
        inactiveNumbers[0] = maxParticles;
        // Populate indices in reverse order so the first element popped off is 0
        for (uint32_t i = 0; i < maxParticles; i++) {
            inactiveNumbers[i + 1] = maxParticles - 1 - i;
        }

        // Map memory
        auto memory = inactiveParticleIndexList.mapCPUMemory(*globalData.context.device);
        memcpy(*memory, inactiveNumbers.data(), sizeof(uint32_t) * inactiveNumbers.size());

        // Map "reset buffer" memory
        auto resetMemory = inactiveParticleIndexList_resetData.mapCPUMemory(*globalData.context.device);
        memcpy(*resetMemory, inactiveNumbers.data(), sizeof(uint32_t) * inactiveNumbers.size());

        // Auto unmap memory
        // Auto unmap reset buffer memory
    }
    inactiveParticleIndexList.scheduleCopyToGPU(*buf);
    inactiveParticleIndexList_resetData.scheduleCopyToGPU(*buf);

    // Zero out other buffers
    buf->fillBuffer(*particlesToEmit, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleDataArray, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleIndexSimulateList, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleIndexDrawList, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleIndirectCommands, 0, VK_WHOLE_SIZE, 0);

    buf->end();

    auto submitInfo = vk::SubmitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*buf;
    globalData.context.computeQueue.submit({submitInfo}, nullptr);

    globalData.context.computeQueue.waitIdle();
}
