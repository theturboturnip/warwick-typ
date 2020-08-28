//
// Created by samuel on 24/08/2020.
//

#include "VulkanPipelineSet.h"

VulkanPipelineSet::VulkanPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<size_t> viewportSize)
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

void VulkanPipelineSet::buildSimulationFragDescriptors(vk::Device device, vk::DescriptorPool pool, VulkanSimulationBuffers buffers) {
    auto allocInfo = vk::DescriptorSetAllocateInfo{};
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &*simulationFragDescriptorLayout;
    simulationFragDescriptors = std::move(device.allocateDescriptorSetsUnique(allocInfo)[0]);

    auto writeDescriptorBuffer = [&](uint32_t i, vk::Buffer buffer, auto allocatedMemory) {
        auto bufferInfo = vk::DescriptorBufferInfo{};
        bufferInfo.buffer = buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = allocatedMemory.totalSize * allocatedMemory.elemSize();

        auto descriptorWrite = vk::WriteDescriptorSet{};
        descriptorWrite.dstSet = *simulationFragDescriptors;
        descriptorWrite.dstBinding = i;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
        descriptorWrite.descriptorCount = 1;

        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;

        device.updateDescriptorSets({descriptorWrite}, {});
    };
    writeDescriptorBuffer(0, buffers.u, buffers.simAllocs.u);
    writeDescriptorBuffer(1, buffers.v, buffers.simAllocs.v);
    writeDescriptorBuffer(2, buffers.p, buffers.simAllocs.p);
    writeDescriptorBuffer(3, buffers.fluidmask, buffers.simAllocs.fluidmask);
}
