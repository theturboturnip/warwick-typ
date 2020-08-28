//
// Created by samuel on 24/08/2020.
//

#pragma once

#include <simulation/memory/vulkan/VulkanSimulationAllocator.h>
#include <vulkan/vulkan.hpp>

#include "VulkanDescriptorSetLayout.h"
#include "VulkanPipeline.h"
#include "VulkanShader.h"

class VulkanPipelineSet {
public:
    struct SimFragPushConstants {
        uint32_t pixelWidth;
        uint32_t pixelHeight;
        uint32_t columnStride;
        uint32_t totalPixels;
    };

private:
    VertexShader triVert;
    VertexShader fullscreenQuadVert;
    FragmentShader redFrag;
    FragmentShader uvFrag;
    FragmentShader simPressure;

    VulkanDescriptorSetLayout simulationFragDescriptorLayout;
    vk::PushConstantRange simulationFragPushConstantRange;

public:
    explicit VulkanPipelineSet(vk::Device device, vk::RenderPass renderPass, Size<uint32_t> viewportSize);
    VulkanPipelineSet(VulkanPipelineSet&&) noexcept = default;

    vk::UniqueDescriptorSet simulationFragDescriptors;
    void buildSimulationFragDescriptors(vk::Device device, vk::DescriptorPool pool, VulkanSimulationBuffers buffers);

    VulkanPipeline redTriangle;
    VulkanPipeline redQuad;
    VulkanPipeline fullscreenPressure;
};
