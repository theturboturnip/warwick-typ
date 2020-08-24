//
// Created by samuel on 24/08/2020.
//
#include "VulkanShader.h"

VulkanShader::VulkanShader(vk::Device device, const std::vector<uint8_t> &data, VulkanShader::Stage stage) {
    auto moduleCreateInfo = vk::ShaderModuleCreateInfo();
    moduleCreateInfo.codeSize = data.size();
    moduleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(data.data());

    shaderModule = device.createShaderModuleUnique(moduleCreateInfo);
    shaderStage = vk::PipelineShaderStageCreateInfo();
    switch (stage){
        case Stage::Vertex:
            shaderStage.stage = vk::ShaderStageFlagBits::eVertex;
        case Stage::Fragment:
            shaderStage.stage = vk::ShaderStageFlagBits::eFragment;
    }
    shaderStage.module = *shaderModule;
    shaderStage.pName = "main";
}

VulkanShader VulkanShader::from_file(vk::Device device, std::string shader_name, VulkanShader::Stage stage) {
    auto shader_file = std::basic_ifstream<uint8_t>("shaders/" + shader_name + ".spv", std::ios::binary);
    auto data = std::vector<uint8_t>((std::istreambuf_iterator<uint8_t>(shader_file)),
                                      std::istreambuf_iterator<uint8_t>());
    return VulkanShader(device, data, stage);
}

