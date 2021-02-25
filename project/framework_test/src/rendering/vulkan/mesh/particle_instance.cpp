//
// Created by samuel on 22/02/2021.
//

#include "particle_instance.h"

#include "vk_format.h"

const vk::VertexInputBindingDescription ParticleInstanceData::bindingDescription(
    1,                               // Binding 1
    sizeof(ParticleInstanceData),            // Stride
    vk::VertexInputRate::eInstance // Per-instance data
);
const std::array<vk::VertexInputAttributeDescription, 2> ParticleInstanceData::attributeDescriptions = {
    vk::VertexInputAttributeDescription(
        2, // Location (NOT relative to the binding)
        1, // Binding
        VulkanFormat<typeof(ParticleInstanceData::data)>::Fmt, // Format
        offsetof(ParticleInstanceData, data) // Offset
    ),
    vk::VertexInputAttributeDescription(
        3, // Location
        1, // Binding
        VulkanFormat<typeof(ParticleInstanceData::color)>::Fmt, // Format
        offsetof(ParticleInstanceData, color) // Offset
    ),
};