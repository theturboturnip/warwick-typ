//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vector>
#include <string>

#include <vulkan/vulkan.hpp>

enum class ShaderStage {
    Vertex,
    Fragment,
    Compute
};

template<ShaderStage Stage>
class VulkanShader {
public:
    std::string name;

    VulkanShader(const VulkanShader&) = delete;
    VulkanShader(VulkanShader&&) noexcept = default;
    static VulkanShader from_file(vk::Device device, std::string shader_name);

    vk::UniqueShaderModule shaderModule;
    vk::PipelineShaderStageCreateInfo getShaderStage(vk::SpecializationInfo* specInfo = nullptr) const;

private:
    VulkanShader(vk::Device device, std::string name, const std::vector<char>& data);
};

using VertexShader = VulkanShader<ShaderStage::Vertex>;
using FragmentShader = VulkanShader<ShaderStage::Fragment>;
using ComputeShader = VulkanShader<ShaderStage::Compute>;