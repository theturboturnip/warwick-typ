//
// Created by samuel on 24/08/2020.
//

#include "VulkanSimPipelineSet.h"

#include "rendering/shaders/global_structures.h"

VulkanSimPipelineSet::VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize)
    : triVert(VertexShader::from_file(device, "triangle.vert")),
      fullscreenQuadVert(VertexShader::from_file(device, "fullscreen_quad.vert")),
      redFrag(FragmentShader::from_file(device, "red.frag")),
      uvFrag(FragmentShader::from_file(device, "uv.frag")),
      simPressure(FragmentShader::from_file(device, "sim_pressure.frag")),
      computeSimDataImage_shader(ComputeShader::from_file(device, "compute_sim_data_image.comp")),

      simBuffers_computeDescriptorLayout(device, {
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
      simBuffersImage_fragmentDescriptorLayout(device, {
          vk::DescriptorSetLayoutBinding(
              0,
              vk::DescriptorType::eCombinedImageSampler,
              1,
              vk::ShaderStageFlagBits::eFragment
          )
      }),
      simBuffers_computePushConstantRange(
              vk::ShaderStageFlagBits::eCompute,
              0,
              sizeof(Shaders::SimDataBufferStats)
              ),

      redTriangle(device, renderPass, viewportSize, triVert, redFrag),
      redQuad(device, renderPass, viewportSize, fullscreenQuadVert, uvFrag),
      fullscreenPressure(device, renderPass, {
              viewportSize.x*2,
              viewportSize.y*2
      }, fullscreenQuadVert, simPressure, &*simBuffersImage_fragmentDescriptorLayout),
      computeSimDataImage(device, computeSimDataImage_shader, &*simBuffers_computeDescriptorLayout, &simBuffers_computePushConstantRange)
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