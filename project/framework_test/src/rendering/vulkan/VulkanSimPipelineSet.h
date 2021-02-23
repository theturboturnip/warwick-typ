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
#include "SimAppProperties.h"

class VulkanSimPipelineSet {
public:
    vk::SpecializationMapEntry particleCount_specConstant;

    VulkanDescriptorSetLayout simDataSampler_comp_ds;
    VulkanDescriptorSetLayout simDataSampler_frag_ds;
    VulkanDescriptorSetLayout simBufferCopyInput_comp_ds;
    VulkanDescriptorSetLayout simBufferCopyOutput_comp_ds;
    VulkanDescriptorSetLayout particleInputBuffer_comp_ds;
    VulkanDescriptorSetLayout particleInputBuffer_vert_ds;
    VulkanDescriptorSetLayout particleOutputBuffer_comp_ds;

    VertexShader quantityScalar_vert;
    FragmentShader quantityScalar_frag;
    VertexShader particle_vert;
    FragmentShader particle_frag;
    ComputeShader computeSimDataImage_shader;
    ComputeShader computeUpdateParticles_shader;

    VulkanPipeline quantityScalar;
    VulkanPipeline particle;
    VulkanPipeline computeSimDataImage;
    VulkanPipeline computeSimUpdateParticles;

    VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize, SimAppProperties& properties);
    VulkanSimPipelineSet(VulkanSimPipelineSet &&) noexcept = default;

    vk::UniqueDescriptorSet buildSimDataSampler_comp_ds(
        VulkanContext& context,
        VulkanImageSampler& simBuffersImageSampler
    );
    vk::UniqueDescriptorSet buildSimDataSampler_frag_ds(
        VulkanContext& context,
        VulkanImageSampler& simBuffersImageSampler
    );
    vk::UniqueDescriptorSet buildSimBufferCopyInput_comp_ds(
        VulkanContext& context,
        VulkanSimFrameData& buffers
    );
    vk::UniqueDescriptorSet buildSimBufferCopyOutput_comp_ds(
        VulkanContext& context,
        VulkanImageSampler& simBuffersImageSampler
    );
    vk::UniqueDescriptorSet buildParticleInputBuffer_comp_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer// TODO is this right?
    );
    vk::UniqueDescriptorSet buildParticleInputBuffer_vert_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer// TODO is this right?
    );
    vk::UniqueDescriptorSet buildParticleOutputBuffer_comp_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer// TODO is this right?
    );
};