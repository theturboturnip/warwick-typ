//
// Created by samuel on 24/08/2020.
//

#include "VulkanPipelineSet.h"

VulkanPipelineSet::VulkanPipelineSet(vk::Device device, Size<size_t> viewportSize)
    : renderPass(),

      triVert(VertexShader::from_file(device, "triangle.vert")),
      redFrag(FragmentShader::from_file(device, "red.frag")),

      redTriangle(device, renderPass, viewportSize, triVert, redFrag)
{}
