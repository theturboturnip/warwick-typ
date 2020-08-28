//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanShader.h"
#include "util/Size.h"

class VulkanPipeline {
    vk::UniquePipelineLayout layout;
public:
    vk::UniquePipeline pipeline;

    VulkanPipeline(vk::Device device, vk::RenderPass renderPass,
                   Size<size_t> viewportSize,
                   const VertexShader& vertex, const FragmentShader& fragment,
                   const vk::DescriptorSetLayout* descriptorSetLayout = nullptr, const vk::PushConstantRange* pushConstantRange = nullptr);
    VulkanPipeline(VulkanPipeline&&) noexcept = default;

    const vk::Pipeline& operator *() const{
        return *pipeline;
    }
};

