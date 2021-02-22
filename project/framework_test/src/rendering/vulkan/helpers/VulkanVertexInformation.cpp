//
// Created by samuel on 22/02/2021.
//

#include <util/fatal_error.h>
#include "VulkanVertexInformation.h"

#include "rendering/vulkan/mesh/vertex.h"
#include "rendering/vulkan/mesh/particle_instance.h"

VulkanVertexInformation VulkanVertexInformation::getInfo(VulkanVertexInformation::Kind kind) {
    switch (kind) {
        case Kind::None:
            return VulkanVertexInformation();
        case Kind::Vertex:
            return VulkanVertexInformation{
                .bindings = { Vertex::bindingDescription },
                .attributes = { Vertex::attributeDescriptions[0], Vertex::attributeDescriptions[1] }
            };
        case Kind::Particle:
            return VulkanVertexInformation{
                .bindings = { Vertex::bindingDescription, ParticleInstanceData::bindingDescription },
                .attributes = {
                    Vertex::attributeDescriptions[0], Vertex::attributeDescriptions[1],
                    ParticleInstanceData::attributeDescriptions[0], ParticleInstanceData::attributeDescriptions[1]
                },
            };
        default:
            FATAL_ERROR("Unhandled Vertex Info Kind");
    }
}