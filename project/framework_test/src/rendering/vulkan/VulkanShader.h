//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vector>
#include <string>

#include <vulkan/vulkan.hpp>

enum class ShaderStage {
    Vertex,
    Fragment
};

template<ShaderStage Stage>
class VulkanShader {
public:
    VulkanShader(const VulkanShader&) = delete;
    VulkanShader(VulkanShader&&) noexcept = default;
    static VulkanShader from_file(vk::Device device, std::string shader_name);

    vk::UniqueShaderModule shaderModule;
    vk::PipelineShaderStageCreateInfo shaderStage;

private:
    VulkanShader(vk::Device device, const std::vector<char>& data);
};

using VertexShader = VulkanShader<ShaderStage::Vertex>;
using FragmentShader = VulkanShader<ShaderStage::Fragment>;