//
// Created by samuel on 24/08/2020.
//

#include "VulkanPipelineSet.h"

VulkanPipelineSet::VulkanPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<size_t> viewportSize)
    : triVert(VertexShader::from_file(device, "triangle.vert")),
      redFrag(FragmentShader::from_file(device, "red.frag")),

      redTriangle(device, renderPass, viewportSize, triVert, redFrag)
{}