//
// Created by samuel on 24/08/2020.
//

#include "VulkanSimPipelineSet.h"

#include "rendering/shaders/global_structures.h"

VulkanSimPipelineSet::VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize, SimAppProperties& props)
    :
    particleCount_specConstant(0, 0, sizeof(uint32_t)),

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
    simBuffers_comp_ds(device, {
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
            ),
            vk::DescriptorSetLayoutBinding(
                    4,
                    vk::DescriptorType::eStorageImage,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            )
    }),
    particleInputBuffer_comp_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            )
    }),
    particleInputBuffer_vert_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eVertex
            )
    }),
    particleOutputBuffer_comp_ds(device, {
            vk::DescriptorSetLayoutBinding(
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute
            )
    }),

    quantityScalar_vert(VertexShader::from_file(device, "quantity_scalar.vert")),
    quantityScalar_frag(FragmentShader::from_file(device, "quantity_scalar.frag")),
    particle_vert(VertexShader::from_file(device, "particle.vert")),
    particle_frag(FragmentShader::from_file(device, "particle.frag")),
    computeSimDataImage_shader(ComputeShader::from_file(device, "compute_sim_data_image.comp")),
    computeUpdateParticles_shader(ComputeShader::from_file(device, "compute_update_particles.comp")),

    quantityScalar(
            device,
            renderPass,
            {
                viewportSize.x*2,
                viewportSize.y*2
            },
            quantityScalar_vert, quantityScalar_frag,
            {*simDataSampler_frag_ds}
    ),
    // TODO This needs vertex inputs
    particle(
            device,
            renderPass,
            {
                    viewportSize.x*2,
                    viewportSize.y*2
            },
            particle_vert, particle_frag,
            {*particleInputBuffer_vert_ds},
            sizeof(Shaders::InstancedParticleParams),
            {
                1,
                &particleCount_specConstant,
                sizeof(uint32_t),
                &props.maxParticles
            }
    ),
    computeSimDataImage(
            device,
            computeSimDataImage_shader,
            {*simBuffers_comp_ds},
            sizeof(Shaders::SimDataBufferStats)
    ),
    computeSimUpdateParticles(
            device,
            computeUpdateParticles_shader,
            {
                *particleInputBuffer_comp_ds,
                *particleOutputBuffer_comp_ds,
                *simDataSampler_comp_ds,
            },
            sizeof(Shaders::ParticleStepParams),
            {
                    1,
                    &particleCount_specConstant,
                    sizeof(uint32_t),
                    &props.maxParticles
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

/*
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimDataSampler_ds(VulkanContext& context, VulkanImageSampler& simBuffersImageSampler) {
    auto allocInfo = vk::DescriptorSetAllocateInfo{};
    allocInfo.descriptorPool = *context.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &*simBuffersImage_fragmentDescriptorLayout;
    auto descriptorSet = std::move(context.device->allocateDescriptorSetsUnique(allocInfo)[0]);

    auto imageInfo = vk::DescriptorImageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = *simBuffersImageSampler.imageView;
    imageInfo.sampler = *simBuffersImageSampler.sampler;

    auto descriptorWrite = vk::WriteDescriptorSet{};
    descriptorWrite.dstSet = *descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptorWrite.descriptorCount = 1;

    descriptorWrite.pBufferInfo = nullptr;
    descriptorWrite.pImageInfo = &imageInfo;
    descriptorWrite.pTexelBufferView = nullptr;

    context.device->updateDescriptorSets({descriptorWrite}, {});
    return descriptorSet;
}
vk::UniqueDescriptorSet
VulkanSimPipelineSet::buildSimBuffersDescriptors(VulkanContext &context, VulkanSimFrameData &buffers, vk::Image simBuffersImage, VulkanImageSampler& simBuffersImageSampler) {
    auto allocInfo = vk::DescriptorSetAllocateInfo{};
    allocInfo.descriptorPool = *context.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &*simBuffers_computeDescriptorLayout;
    auto descriptorSet = std::move(context.device->allocateDescriptorSetsUnique(allocInfo)[0]);

    auto writeDescriptorBuffer = [&](uint32_t i, vk::DescriptorBufferInfo bufferInfo) {
        auto descriptorWrite = vk::WriteDescriptorSet{};
        descriptorWrite.dstSet = *descriptorSet;
        descriptorWrite.dstBinding = i;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
        descriptorWrite.descriptorCount = 1;

        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;

        context.device->updateDescriptorSets({descriptorWrite}, {});
    };
    writeDescriptorBuffer(0, buffers.u);
    writeDescriptorBuffer(1, buffers.v);
    writeDescriptorBuffer(2, buffers.p);
    writeDescriptorBuffer(3, buffers.fluidmask);

    {
        auto imageInfo = vk::DescriptorImageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eGeneral;
        imageInfo.imageView = *simBuffersImageSampler.imageView;
        imageInfo.sampler = *simBuffersImageSampler.sampler;

        auto descriptorWrite = vk::WriteDescriptorSet{};
        descriptorWrite.dstSet = *descriptorSet;
        descriptorWrite.dstBinding = 4;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eStorageImage;
        descriptorWrite.descriptorCount = 1;

        descriptorWrite.pBufferInfo = nullptr;
        descriptorWrite.pImageInfo = &imageInfo;
        descriptorWrite.pTexelBufferView = nullptr;

        context.device->updateDescriptorSets({descriptorWrite}, {});
    }

    return descriptorSet;
}
 */

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
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimBuffers_comp_ds(
        VulkanContext& context,
        VulkanSimFrameData& buffers,
        VulkanImageSampler& simBuffersImageSampler
){
    return buildDescriptorSet(
        context,
        simBuffers_comp_ds,
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
            },
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
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildParticleInputBuffer_comp_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
){
    return buildDescriptorSet(
        context,
        particleInputBuffer_comp_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffer,
                .imageInfo = std::nullopt
            },
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildParticleInputBuffer_vert_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
){
    return buildDescriptorSet(
        context,
        particleInputBuffer_vert_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffer,
                .imageInfo = std::nullopt
            },
        }
    );
}
vk::UniqueDescriptorSet VulkanSimPipelineSet::buildParticleOutputBuffer_comp_ds(
        VulkanContext& context,
        vk::DescriptorBufferInfo buffer
){
    return buildDescriptorSet(
        context,
        particleOutputBuffer_comp_ds,
        {
            Descriptor{
                .type = vk::DescriptorType::eStorageBuffer,
                .bufferInfo = buffer,
                .imageInfo = std::nullopt
            },
        }
    );
}