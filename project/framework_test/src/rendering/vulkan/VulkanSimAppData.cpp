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

      particleEmitters(context, vk::BufferUsageFlagBits::eStorageBuffer, globalData.props.maxParticleEmitters * sizeof(Shaders::ParticleEmitter)),
      particleEmitters_comp_ds(bufferDescSet_comp(globalData, particleEmitters.getGpuDescriptor())),

      quantityScalar_range(context, vk::BufferUsageFlagBits::eStorageBuffer, sizeof(Shaders::FloatRange)),
      quantityScalar_range_frag_ds(bufferDescSet_frag(globalData, quantityScalar_range.getGpuDescriptor())),
      quantityVector_range(context, vk::BufferUsageFlagBits::eStorageBuffer, sizeof(Shaders::FloatRange)),
      quantityVector_range_comp_ds(bufferDescSet_comp(globalData, quantityVector_range.getGpuDescriptor())),

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

        quantityScalar(context, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled, simDataImage.size, vk::Format::eR32G32Sfloat, true),
        quantityScalarSampler(context, quantityScalar),
        quantityScalar_comp_ds(globalData.pipelines.buildImage_comp_ds(context, quantityScalarSampler)),
        quantityScalarSampler_frag_ds(globalData.pipelines.buildImageSampler_frag_ds(context, quantityScalarSampler)),
        quantityScalarReducer(context, globalData.pipelines, simDataImage.size.area()),

        quantityVector(context, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled, simDataImage.size, vk::Format::eR32G32B32A32Sfloat, true),
        quantityVectorSampler(context, quantityVector),
        quantityVector_comp_ds(globalData.pipelines.buildImage_comp_ds(context, quantityVectorSampler)),
        quantityVectorSampler_comp_ds(globalData.pipelines.buildImageSampler_comp_ds(context, quantityVectorSampler)),
        quantityVectorSampler_frag_ds(globalData.pipelines.buildImageSampler_frag_ds(context, quantityVectorSampler)),
        quantityVectorReducer(context, globalData.pipelines, simDataImage.size.area()),
        quantityVectorIndirectDrawData(
            context,
            vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            sizeof(Shaders::VectorArrowIndirectCommands),
            false
        ),
        quantityVectorIndirectDrawData_comp_ds(bufferDescSet_comp(globalData, quantityVectorIndirectDrawData.getGpuDescriptor())),
        vectorArrowInstanceData(
            context,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            (sizeof(Shaders::VectorArrow) * globalData.props.maxVectorArrows) + sizeof(uint32_t),
            false
        ),
        vectorArrowInstanceData_comp_ds(bufferDescSet_comp(globalData, vectorArrowInstanceData.asDescriptor())),
        vectorArrowInstanceData_vert_ds(bufferDescSet_vert(globalData, vectorArrowInstanceData.asDescriptor())),
        vectorArrowVertexIndexData(
            context,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
            sizeof(Vertex) * 7 + sizeof(uint16_t) * 8,
            false // not shared
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
{
    // Alloc a command buffer to init data (auto deleted)
    vk::UniqueCommandBuffer buf = std::move(globalData.context.allocateCommandBuffers(globalData.context.computeCmdPool, vk::CommandBufferLevel::ePrimary, 1)[0]);

    const auto beginInfo = vk::CommandBufferBeginInfo{};
    buf->begin(&beginInfo);

    // Initialize particle vertex buffer
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

    // Initialize inactive particle list
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

    // Initialize the vector arrow vertex+index buffer
    {
        auto memory = vectorArrowVertexIndexData.mapCPUMemory(*globalData.context.device);
        auto* vData = (Vertex*)(*memory);
        vData[0] = Vertex{
                .pos = glm::vec2(-1, 0.16),
                .uv = glm::vec2(0, 0),
        };
        vData[1] = Vertex{
                .pos = glm::vec2(-1, -0.16),
                .uv = glm::vec2(0, 0),
        };
        vData[2] = Vertex{
                .pos = glm::vec2(0.25, 0.16),
                .uv = glm::vec2(0, 0),
        };
        vData[3] = Vertex{
                .pos = glm::vec2(0.25, -0.16),
                .uv = glm::vec2(0, 0),
        };
        vData[4] = Vertex{
                .pos = glm::vec2(0.25, 0.75),
                .uv = glm::vec2(0, 0),
        };
        vData[5] = Vertex{
                .pos = glm::vec2(0.25, -0.75),
                .uv = glm::vec2(0, 0),
        };
        vData[6] = Vertex{
                .pos = glm::vec2(1, 0),
                .uv = glm::vec2(0, 0),
        };
        auto* idxData = (uint16_t*)(vData + 7);
        idxData[0] = 0;
        idxData[1] = 1;
        idxData[2] = 2;
        idxData[3] = 3;
        idxData[4] = 0xFFFF;
        idxData[5] = 4;
        idxData[6] = 5;
        idxData[7] = 6;

        // Auto unmapped
    }
    vectorArrowVertexIndexData.scheduleCopyToGPU(*buf);

    // initialize the quantityVectorIndirectDrawData
    {
        auto memory = quantityVectorIndirectDrawData.mapCPUMemory(*globalData.context.device);
        auto* data = (Shaders::VectorArrowIndirectCommands*)(*memory);
        data->vectorArrowDrawCmd = Shaders::VkDrawIndexedIndirectCommand{
            .indexCount = 8,
            .instanceCount = 0, // Starts at 0, is incremented by computeVectorArrowGenerate shader
            .firstIndex = 0,
            .vertexOffset = 0,
            .firstInstance = 0,
        };
        // Auto unmapped
    }
    quantityVectorIndirectDrawData.scheduleCopyToGPU(*buf);

    // Zero out other buffers
    buf->fillBuffer(*particlesToEmit, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleDataArray, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleIndexSimulateList, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleIndexDrawList, 0, VK_WHOLE_SIZE, 0);
    buf->fillBuffer(*particleIndirectCommands, 0, VK_WHOLE_SIZE, 0);

    buf->fillBuffer(*vectorArrowInstanceData, 0, VK_WHOLE_SIZE, 0);


    buf->end();

    auto submitInfo = vk::SubmitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*buf;
    globalData.context.computeQueue.submit({submitInfo}, nullptr);

    globalData.context.computeQueue.waitIdle();
}
