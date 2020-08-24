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

    vk::RenderPass renderPass;

public:
    explicit VulkanPipelineSet(vk::Device device, Size<size_t> viewportSize);

    VulkanPipeline redTriangle;
};
