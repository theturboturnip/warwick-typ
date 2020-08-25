//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include "VulkanPipeline.h"
#include "VulkanShader.h"

class VulkanPipelineSet {
    VertexShader triVert;
    FragmentShader redFrag;
public:
    explicit VulkanPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<size_t> viewportSize);

    VulkanPipeline redTriangle;
};
