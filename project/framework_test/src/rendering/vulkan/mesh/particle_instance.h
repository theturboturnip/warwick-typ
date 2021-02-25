//
// Created by samuel on 22/02/2021.
//

#pragma once

#include <array>
#include <vulkan/vulkan.hpp>

#include "rendering/shaders/global_structures.h"

// Helper struct for handling particle instance data
struct ParticleInstanceData : public Shaders::Particle {

    inline glm::vec2 pos() {
        return glm::vec2(data.x, data.y);
    }
    inline float rot() {
        return data.z;
    }


    static const vk::VertexInputBindingDescription bindingDescription;
    static const std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;
};