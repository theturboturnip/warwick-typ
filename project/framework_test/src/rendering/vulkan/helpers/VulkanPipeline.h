//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanVertexInformation.h"
#include "VulkanShader.h"
#include "util/Size.h"

class VulkanPipeline {
public:
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::PushConstantRange pushConstantRange;

    VulkanPipeline(vk::Device device, vk::RenderPass renderPass,
                   Size<uint32_t> viewportSize,
                   const VertexShader& vertex, const FragmentShader& fragment,
                   VulkanVertexInformation::Kind vertexInfoKind,
                   const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts = {},
                   size_t pushConstantSize=0,
                   vk::SpecializationInfo specInfo={0, nullptr, 0, nullptr});
    VulkanPipeline(vk::Device device,
                   const ComputeShader& compute,
                   const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts = {},
                   size_t pushConstantSize=0,
                   vk::SpecializationInfo specInfo={0, nullptr, 0, nullptr});
    VulkanPipeline(VulkanPipeline&&) noexcept = default;

    const vk::Pipeline& operator *() const{
        return *pipeline;
    }
};

