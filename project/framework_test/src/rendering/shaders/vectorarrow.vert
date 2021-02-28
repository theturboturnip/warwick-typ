#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_descriptor_sets.glsl"
#include "global_structures.glsl"

SPEC_CONST_MAX_VECTORARROW_COUNT()

PUSH_CONSTANTS(InstancedVectorArrowParams)

layout(set = 0, binding = 0, std430) readonly buffer VectorArrowIndices {
    uint vectorArrowDatas_length;
    VectorArrow vectorArrowDatas[maxVectorArrowCount];
};

// Per-vertex data
layout(location = 0) in vec2 v_pos;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec4 f_color;

void main() {
    const VectorArrow vectorArrow = vectorArrowDatas[gl_InstanceIndex.x];

    vec2 pos_01space = (vectorArrow.rotScale * v_pos.xy) + vectorArrow.pos;
    vec2 finalPos = (pos_01space * 2) - 1;
    gl_Position = vec4(finalPos.x, finalPos.y, 0, 1);
    f_uv = v_uv;
    // TODO - colorramp based on quantity?
    f_color = vec4(1,0,0,1);
}