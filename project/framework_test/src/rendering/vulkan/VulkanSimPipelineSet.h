//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <memory/FrameSetAllocator.h>

#include "rendering/vulkan/helpers/VulkanDescriptorSetLayout.h"
#include "rendering/vulkan/helpers/VulkanPipeline.h"
#include "rendering/vulkan/helpers/VulkanShader.h"
#include "rendering/vulkan/VulkanContext.h"

class VulkanSimPipelineSet {
public:
    struct SimFragPushConstants {
        uint32_t pixelWidth;
        uint32_t pixelHeight;
        uint32_t columnStride;
        uint32_t totalPixels;
    };

    VertexShader triVert;
    VertexShader fullscreenQuadVert;
    FragmentShader redFrag;
    FragmentShader uvFrag;
    FragmentShader simPressure;

    VulkanDescriptorSetLayout simulationFragDescriptorLayout;
    vk::PushConstantRange simulationFragPushConstantRange;

    VulkanPipeline redTriangle;
    VulkanPipeline redQuad;
    VulkanPipeline fullscreenPressure;

    explicit VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize);
    VulkanSimPipelineSet(VulkanSimPipelineSet &&) noexcept = default;

    vk::UniqueDescriptorSet buildSimulationFragDescriptors(VulkanContext& context, VulkanSimFrameData& buffers);
};