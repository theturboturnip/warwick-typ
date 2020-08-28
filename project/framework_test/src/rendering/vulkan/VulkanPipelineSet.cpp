//
// Created by samuel on 24/08/2020.
//

#include "VulkanPipelineSet.h"

VulkanPipelineSet::VulkanPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<size_t> viewportSize)
    : triVert(VertexShader::from_file(device, "triangle.vert")),
      fullscreenQuadVert(VertexShader::from_file(device, "fullscreen_quad.vert")),
      redFrag(FragmentShader::from_file(device, "red.frag")),
      simPressure(FragmentShader::from_file(device, "sim_pressure.frag")),

      redTriangle(device, renderPass, viewportSize, triVert, redFrag),
      redQuad(device, renderPass, viewportSize, fullscreenQuadVert, redFrag),
      fullscreenPressure(device, renderPass, viewportSize, fullscreenQuadVert, simPressure)
{}