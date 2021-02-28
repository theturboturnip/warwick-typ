//
// Created by samuel on 24/08/2020.
//
#include "VulkanShader.h"
#include <util/fatal_error.h>

template<ShaderStage Stage>
VulkanShader<Stage>::VulkanShader(vk::Device device, std::string name, const std::vector<char> &data) : name(name) {
    auto moduleCreateInfo = vk::ShaderModuleCreateInfo();
    moduleCreateInfo.codeSize = data.size();
    moduleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(data.data());

    shaderModule = device.createShaderModuleUnique(moduleCreateInfo);
}

template<ShaderStage Stage>
VulkanShader<Stage> VulkanShader<Stage>::from_file(vk::Device device, std::string shader_name) {
    auto shader_file = std::basic_ifstream<char>("shaders/" + shader_name + ".spv", std::ios::binary);
    auto data = std::vector<char>((std::istreambuf_iterator<char>(shader_file)),
                                      std::istreambuf_iterator<char>());
    FATAL_ERROR_IF(data.size() == 0, "No SPIRV found in %s", shader_name.c_str());
    FATAL_ERROR_IF(data.size() % 4 != 0, "Vulkan expects SPIR-V files to be multiples of uint32 i.e. have a size that's a multiple of 4.");
    return VulkanShader<Stage>(device, shader_name, data);
}

template<ShaderStage Stage>
vk::PipelineShaderStageCreateInfo VulkanShader<Stage>::getShaderStage(vk::SpecializationInfo* info) const {
    auto shaderStage = vk::PipelineShaderStageCreateInfo();
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

    if (info && info->dataSize > 0) {
        shaderStage.pSpecializationInfo = info;
    } else {
        shaderStage.pSpecializationInfo = nullptr;
    }

    shaderStage.pName = "main";
    return shaderStage;
}

template class VulkanShader<ShaderStage::Vertex>;
template class VulkanShader<ShaderStage::Fragment>;
template class VulkanShader<ShaderStage::Compute>;
