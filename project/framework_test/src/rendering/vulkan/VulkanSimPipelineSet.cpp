//
// Created by samuel on 24/08/2020.
//

#include "VulkanSimPipelineSet.h"

VulkanSimPipelineSet::VulkanSimPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize)
    : triVert(VertexShader::from_file(device, "triangle.vert")),
      fullscreenQuadVert(VertexShader::from_file(device, "fullscreen_quad.vert")),
      redFrag(FragmentShader::from_file(device, "red.frag")),
      uvFrag(FragmentShader::from_file(device, "uv.frag")),
      simPressure(FragmentShader::from_file(device, "sim_pressure.frag")),

      simulationFragDescriptorLayout(device, {
                                                  vk::DescriptorSetLayoutBinding(
                                                             0,
                                                             vk::DescriptorType::eStorageBuffer,
                                                             1,
                                                             vk::ShaderStageFlagBits::eFragment
                                                             ),
                                                  vk::DescriptorSetLayoutBinding(
                                                          1,
                                                          vk::DescriptorType::eStorageBuffer,
                                                          1,
                                                          vk::ShaderStageFlagBits::eFragment
                                                  ),
                                                  vk::DescriptorSetLayoutBinding(
                                                          2,
                                                          vk::DescriptorType::eStorageBuffer,
                                                          1,
                                                          vk::ShaderStageFlagBits::eFragment
                                                  ),
                                                  vk::DescriptorSetLayoutBinding(
                                                          3,
                                                          vk::DescriptorType::eStorageBuffer,
                                                          1,
                                                          vk::ShaderStageFlagBits::eFragment
                                                  ),
                                             }),
      simulationFragPushConstantRange(
              vk::ShaderStageFlagBits::eFragment,
              0,
              sizeof(SimFragPushConstants)
              ),

      redTriangle(device, renderPass, viewportSize, triVert, redFrag),
      redQuad(device, renderPass, viewportSize, fullscreenQuadVert, uvFrag),
      fullscreenPressure(device, renderPass, viewportSize, fullscreenQuadVert, simPressure, &*simulationFragDescriptorLayout, &simulationFragPushConstantRange)
{}

vk::UniqueDescriptorSet VulkanSimPipelineSet::buildSimulationFragDescriptors(VulkanContext& context, VulkanSimFrameData& buffers) {
    auto allocInfo = vk::DescriptorSetAllocateInfo{};
    allocInfo.descriptorPool = *context.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &*simulationFragDescriptorLayout;
    auto simulationFragDescriptorSet = std::move(context.device->allocateDescriptorSetsUnique(allocInfo)[0]);

    auto writeDescriptorBuffer = [&](uint32_t i, vk::DescriptorBufferInfo bufferInfo) {
        auto descriptorWrite = vk::WriteDescriptorSet{};
        descriptorWrite.dstSet = *simulationFragDescriptorSet;
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

    return simulationFragDescriptorSet;
}
