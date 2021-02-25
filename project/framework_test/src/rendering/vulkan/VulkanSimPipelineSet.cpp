//
// Created by samuel on 24/08/2020.
//

#include "VulkanSimPipelineSet.h"

#include "rendering/shaders/global_structures.h"

VulkanSimPipelineSet::VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize, SimAppProperties& props)
    :
        particleBufferLength_specConstant(0, 0, sizeof(uint32_t)),
        particleToEmitBufferLength_specConstant(1, sizeof(uint32_t), sizeof(uint32_t)),
        particleEmitterCount_specConstant(2, sizeof(uint32_t) * 2, sizeof(uint32_t)),
        specConstants({
            particleBufferLength_specConstant,
            particleToEmitBufferLength_specConstant,
            particleEmitterCount_specConstant
        }),
        specConstantsData({
            .particleBufferLength=props.maxParticles,
            .particleToEmitBufferLength=props.maxParticlesEmittedPerFrame,
            .particleEmitterCount=props.maxParicleEmitters
        }),

    simDataSampler_comp_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eCombinedImageSampler,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            )
    }),
    simDataSampler_frag_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eCombinedImageSampler,
                    1,
                    vk::ShaderStageFlagBits::eFragment
            )
    }),
    simBufferCopyInput_comp_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            ),
            vk::DescriptorSetLayoutBinding(
                    1,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            ),
            vk::DescriptorSetLayoutBinding(
                    2,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            ),
            vk::DescriptorSetLayoutBinding(
                    3,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            )
    }),
    simBufferCopyOutput_comp_ds(device, {
        vk::DescriptorSetLayoutBinding(
            0,
            vk::DescriptorType::eStorageImage,
            1,
            vk::ShaderStageFlagBits::eCompute
        )
    }),
    buffer_comp_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            )
    }),
    buffer_vert_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eVertex
            )
    }),
    buffer_frag_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eFragment
            )
    }),

    quantityScalar_vert(VertexShader::from_file(device, "quantity_scalar.vert")),
    quantityScalar_frag(FragmentShader::from_file(device, "quantity_scalar.frag")),
    particle_vert(VertexShader::from_file(device, "particle.vert")),
    particle_frag(FragmentShader::from_file(device, "particle.frag")),
    computeSimDataImage_shader(ComputeShader::from_file(device, "compute_sim_data_image.comp")),
    computeParticleKickoff_shader(ComputeShader::from_file(device, "compute_particle_kickoff.comp")),
    computeParticleEmit_shader(ComputeShader::from_file(device, "compute_particle_emit.comp")),
    computeParticleSimulate_shader(ComputeShader::from_file(device, "compute_particle_simulate.comp")),

    quantityScalar(
            {ScalarQuantity::None, ScalarQuantity::VelocityX, ScalarQuantity::VelocityY, ScalarQuantity::VelocityMagnitude, ScalarQuantity::Pressure, ScalarQuantity::Vorticity},
            device,
            renderPass,
            {
                viewportSize.x*2,
                viewportSize.y*2
            },
            quantityScalar_vert, quantityScalar_frag,
            VulkanVertexInformation::Kind::None,
            {*simDataSampler_frag_ds, *buffer_frag_ds},
            sizeof(Shaders::QuantityScalarParams)
    ),
    particle(
            device,
            renderPass,
            {
                    viewportSize.x*2,
                    viewportSize.y*2
            },
            particle_vert, particle_frag,
            VulkanVertexInformation::Kind::Vertex,
            {
                *buffer_vert_ds,
                *buffer_vert_ds,
            },
            sizeof(Shaders::InstancedParticleParams)
    ),
    computeSimDataImage(
            device,
            computeSimDataImage_shader,
            {*simBufferCopyInput_comp_ds, *simBufferCopyOutput_comp_ds},
            sizeof(Shaders::SimDataBufferStats)
    ),
    computeParticleKickoff(
            device,
            computeParticleKickoff_shader,
            {
                *buffer_comp_ds,
                *buffer_comp_ds,
                *buffer_comp_ds,
            },
            sizeof(Shaders::ParticleKickoffParams),
            {
                    (uint32_t)specConstants.size(),
                    specConstants.data(),
                    sizeof(specConstantsData),
                    &specConstantsData
            }
    ),
    computeParticleEmit(
            device,
            computeParticleEmit_shader,
            {
                    *buffer_comp_ds,
                    *buffer_comp_ds,
                    *buffer_comp_ds,
                    *buffer_comp_ds,
                    *buffer_comp_ds,
            },
            0,
            {
                    (uint32_t)specConstants.size(),
                    specConstants.data(),
                    sizeof(specConstantsData),
                    &specConstantsData
            }
    ),
    computeParticleSimulate(
            device,
            computeParticleSimulate_shader,
            {
                    *buffer_comp_ds,
                    *simDataSampler_comp_ds,
                    *buffer_comp_ds,
                    *buffer_comp_ds,
                    *buffer_comp_ds,
                    *buffer_comp_ds,
            },
            sizeof(Shaders::ParticleSimulateParams),
            {
                    (uint32_t)specConstants.size(),
                    specConstants.data(),
                    sizeof(specConstantsData),
                    &specConstantsData
            }
    )
{}

struct Descriptor {
    vk::DescriptorType type;
    std::optional<vk::DescriptorBufferInfo> bufferInfo;
    std::optional<vk::DescriptorImageInfo> imageInfo;
};

vk::UniqueDescriptorSet buildDescriptorSet(VulkanContext& context,
                                           VulkanDescriptorSetLayout& layout,
                                           const std::vector<Descriptor>& descriptors) {
    auto allocInfo = vk::DescriptorSetAllocateInfo{};
    allocInfo.descriptorPool = *context.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &(*layout);
    auto descriptorSet = std::move(context.device->allocateDescriptorSetsUnique(allocInfo)[0]);

    auto writes = std::vector<vk::WriteDescriptorSet>();
    size_t i = 0;
    for (const auto& descriptor : descriptors) {
        auto descriptorWrite = vk::WriteDescriptorSet{};
        descriptorWrite.dstSet = *descriptorSet;
        descriptorWrite.dstBinding = i;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = descriptor.type;
        descriptorWrite.descriptorCount = 1;

        descriptorWrite.pBufferInfo = descriptor.bufferInfo.has_value() ? &descriptor.bufferInfo.value() : nullptr;
        descriptorWrite.pImageInfo = descriptor.imageInfo.has_value() ? &descriptor.imageInfo.value() : nullptr;
        descriptorWrite.pTexelBufferView = nullptr;
        writes.push_back(descriptorWrite);
        i++;
    }

    context.device->updateDescriptorSets(writes, {});
    return descriptorSet;
}

vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimDataSampler_comp_ds(
        VulkanContext& context,
        VulkanImageSampler& simBuffersImageSampler
){
    return buildDescriptorSet(
        context,
        simDataSampler_comp_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .bufferInfo = std::nullopt,
                .imageInfo = vk::DescriptorImageInfo(
                    *simBuffersImageSampler.sampler,
                    *simBuffersImageSampler.imageView,
                    vk::ImageLayout::eShaderReadOnlyOptimal
                )
            }
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimDataSampler_frag_ds(
        VulkanContext& context,
        VulkanImageSampler& simBuffersImageSampler
){
    return buildDescriptorSet(
        context,
        simDataSampler_frag_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .bufferInfo = std::nullopt,
                .imageInfo = vk::DescriptorImageInfo(
                    *simBuffersImageSampler.sampler,
                    *simBuffersImageSampler.imageView,
                    vk::ImageLayout::eShaderReadOnlyOptimal
                )
            }
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimBufferCopyInput_comp_ds(
        VulkanContext& context,
        VulkanSimFrameData& buffers
){
    return buildDescriptorSet(
        context,
        simBufferCopyInput_comp_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffers.u,
                .imageInfo = std::nullopt
            },
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffers.v,
                .imageInfo = std::nullopt
            },
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffers.p,
                .imageInfo = std::nullopt
            },
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffers.fluidmask,
                .imageInfo = std::nullopt
            }
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimBufferCopyOutput_comp_ds(
        VulkanContext& context,
        VulkanImageSampler& simBuffersImageSampler
){
    return buildDescriptorSet(
            context,
            simBufferCopyOutput_comp_ds,
            {
                    Descriptor{
                            .type = vk::DescriptorType::eStorageImage,
                            .bufferInfo = std::nullopt,
                            .imageInfo = vk::DescriptorImageInfo(
                                    nullptr, // No sampler, this image isn't being sampled
                                    *simBuffersImageSampler.imageView,
                                    vk::ImageLayout::eGeneral
                            )
                    }
            }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildBuffer_comp_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
){
    return buildDescriptorSet(
        context,
        buffer_comp_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffer,
                .imageInfo = std::nullopt
            },
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildBuffer_vert_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
){
    return buildDescriptorSet(
        context,
        buffer_vert_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffer,
                .imageInfo = std::nullopt
            },
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildBuffer_frag_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
){
    return buildDescriptorSet(
            context,
            buffer_frag_ds,
            {
                    Descriptor{
                            .type = vk::DescriptorType::eStorageBuffer,
                            .bufferInfo = buffer,
                            .imageInfo = std::nullopt
                    },
            }
    );
}
