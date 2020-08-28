//
// Created by samuel on 26/08/2020.
//

#pragma once

#include <vulkan/vulkan.hpp>

class VulkanRenderPass {
    vk::UniqueRenderPass renderPass;

public:
    enum class Position {
        PipelineStart,
        PipelineMiddle,
        PipelineEnd,
        PipelineStartAndEnd
    };

    // TODO - AFAIK this uses the copy ctor. Why does that work?
    VulkanRenderPass() : renderPass(nullptr) {}
    VulkanRenderPass(vk::Device device, vk::Format surfaceFormat, Position position, vk::ImageLayout targetLayout);

    const vk::RenderPass& operator *() const {
        return *renderPass;
    }
};

