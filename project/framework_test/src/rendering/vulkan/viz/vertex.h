//
// Created by samuel on 22/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include "util/glm.h"

struct Vertex {
    glm::vec2 pos;
    glm::vec2 uv;

    static const vk::VertexInputBindingDescription bindingDescription;
    static const std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;
};