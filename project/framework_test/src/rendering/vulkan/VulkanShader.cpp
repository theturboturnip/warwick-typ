//
// Created by samuel on 24/08/2020.
//
#include "VulkanShader.h"

template<ShaderStage Stage>
VulkanShader<Stage>::VulkanShader(vk::Device device, const std::vector<uint8_t> &data) {
    auto moduleCreateInfo = vk::ShaderModuleCreateInfo();
    moduleCreateInfo.codeSize = data.size();
    moduleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(data.data());

    shaderModule = device.createShaderModuleUnique(moduleCreateInfo);
    shaderStage = vk::PipelineShaderStageCreateInfo();
    switch (Stage){
        case ShaderStage::Vertex:
            shaderStage.stage = vk::ShaderStageFlagBits::eVertex;
            break;
        case ShaderStage::Fragment:
            shaderStage.stage = vk::ShaderStageFlagBits::eFragment;
            break;
    }
    shaderStage.module = *shaderModule;
    shaderStage.pName = "main";
}

template<ShaderStage Stage>
VulkanShader<Stage> VulkanShader<Stage>::from_file(vk::Device device, std::string shader_name) {
    auto shader_file = std::basic_ifstream<uint8_t>("shaders/" + shader_name + ".spv", std::ios::binary);
    auto data = std::vector<uint8_t>((std::istreambuf_iterator<uint8_t>(shader_file)),
                                      std::istreambuf_iterator<uint8_t>());
    return VulkanShader<Stage>(device, data);
}

template class VulkanShader<ShaderStage::Vertex>;
template class VulkanShader<ShaderStage::Fragment>;
