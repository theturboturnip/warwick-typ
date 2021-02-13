//
// Created by samuel on 24/08/2020.
//
#include "VulkanShader.h"
#include <util/fatal_error.h>

template<ShaderStage Stage>
VulkanShader<Stage>::VulkanShader(vk::Device device, const std::vector<char> &data) {
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
        case ShaderStage::Compute:
            shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
            break;
    }
    shaderStage.module = *shaderModule;
    shaderStage.pName = "main";
}

template<ShaderStage Stage>
VulkanShader<Stage> VulkanShader<Stage>::from_file(vk::Device device, std::string shader_name) {
    auto shader_file = std::basic_ifstream<char>("shaders/" + shader_name + ".spv", std::ios::binary);
    auto data = std::vector<char>((std::istreambuf_iterator<char>(shader_file)),
                                      std::istreambuf_iterator<char>());
    FATAL_ERROR_IF(data.size() % 4 != 0, "Vulkan expects SPIR-V files to be multiples of uint32 i.e. have a size that's a multiple of 4.");
    return VulkanShader<Stage>(device, data);
}

template class VulkanShader<ShaderStage::Vertex>;
template class VulkanShader<ShaderStage::Fragment>;
template class VulkanShader<ShaderStage::Compute>;
