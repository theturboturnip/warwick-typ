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
    vk::SpecializationMapEntry particleBufferLength_specConstant;
    vk::SpecializationMapEntry particleToEmitBufferLength_specConstant;
    vk::SpecializationMapEntry particleEmitterCount_specConstant;
    std::vector<vk::SpecializationMapEntry> specConstants;
    struct SpecConstantsData {
        uint32_t particleBufferLength;
        uint32_t particleToEmitBufferLength;
        uint32_t particleEmitterCount;
    } specConstantsData;

    VulkanDescriptorSetLayout simDataSampler_comp_ds;
    VulkanDescriptorSetLayout simDataSampler_frag_ds;
    VulkanDescriptorSetLayout simBufferCopyInput_comp_ds;
    VulkanDescriptorSetLayout simBufferCopyOutput_comp_ds;

    VulkanDescriptorSetLayout buffer_comp_ds;
    VulkanDescriptorSetLayout buffer_vert_ds;

    VertexShader quantityScalar_vert;
    FragmentShader quantityScalar_frag;
    VertexShader particle_vert;
    FragmentShader particle_frag;
    ComputeShader computeSimDataImage_shader;

    ComputeShader computeParticleKickoff_shader;
    ComputeShader computeParticleEmit_shader;
    ComputeShader computeParticleSimulate_shader;

    VulkanPipeline quantityScalar;
    VulkanPipeline particle;
    VulkanPipeline computeSimDataImage;
    VulkanPipeline computeParticleKickoff;
    VulkanPipeline computeParticleEmit;
    VulkanPipeline computeParticleSimulate;

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
    vk::UniqueDescriptorSet buildBuffer_comp_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
    );
    vk::UniqueDescriptorSet buildBuffer_vert_ds(
            VulkanContext& context,
            vk::DescriptorBufferInfo buffer
    );
};