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
    particle(
            device,
            renderPass,
            {
                    viewportSize.x*2,
                    viewportSize.y*2
            },
            particle_vert, particle_frag,
            {*simDataSampler_frag_ds},
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