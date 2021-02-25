//
// Created by samuel on 22/02/2021.
//

#include "vertex.h"

#include "vk_format.h"

const vk::VertexInputBindingDescription Vertex::bindingDescription(
        0,                             // Binding
        sizeof(Vertex),                       // Stride
        vk::VertexInputRate::eVertex // Per-vertex data
);
const std::array<vk::VertexInputAttributeDescription, 2> Vertex::attributeDescriptions = {
        vk::VertexInputAttributeDescription(
                0, // Location
                0, // Binding
                VulkanFormat<typeof(Vertex::pos)>::Fmt, // Format
                offsetof(Vertex, pos) // Offset
        ),
        vk::VertexInputAttributeDescription(
                1, // Location
                0, // Binding
                VulkanFormat<typeof(Vertex::uv)>::Fmt, // Format
                offsetof(Vertex, uv) // Offset
        ),
};