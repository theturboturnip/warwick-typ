//
// Created by samuel on 22/02/2021.
//

#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>

struct VulkanVertexInformation {
    enum class Kind {
        None,
        Vertex,
    };

    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;

    static VulkanVertexInformation getInfo(Kind kind);
};
