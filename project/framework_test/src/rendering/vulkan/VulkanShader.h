//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vector>
#include <string>

#include <vulkan/vulkan.hpp>

// TODO - make Stage a template parameter? could be nice to ensure type safety
class VulkanShader {
public:
    enum class Stage {
        Vertex,
        Fragment
    };

    static VulkanShader from_file(vk::Device device, std::string shader_name, Stage stage);

    vk::UniqueShaderModule shaderModule;
    vk::PipelineShaderStageCreateInfo shaderStage;

private:
    VulkanShader(vk::Device device, const std::vector<uint8_t>& data, Stage stage);
};