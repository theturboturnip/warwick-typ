//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <memory/FrameSetAllocator.h>
#include <rendering/vulkan/helpers/VulkanImageSampler.h>

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
    ComputeShader computeSimDataImage_shader;

    VulkanDescriptorSetLayout simBuffers_computeDescriptorLayout;
    VulkanDescriptorSetLayout simBuffersImage_fragmentDescriptorLayout;
    vk::PushConstantRange simBuffers_computePushConstantRange;

    VulkanPipeline redTriangle;
    VulkanPipeline redQuad;
    VulkanPipeline fullscreenPressure;
    VulkanPipeline computeSimDataImage;

    explicit VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize);
    VulkanSimPipelineSet(VulkanSimPipelineSet &&) noexcept = default;

    vk::UniqueDescriptorSet buildFullscreenPressureDescriptors(VulkanContext& context, VulkanImageSampler& simBuffersImageSampler);
    vk::UniqueDescriptorSet buildComputeSimDataImageDescriptors(VulkanContext& context, VulkanSimFrameData& buffers, vk::Image simBuffersImage, VulkanImageSampler& simBuffersImageSampler);
};