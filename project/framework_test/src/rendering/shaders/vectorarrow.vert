#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_descriptor_sets.glsl"
#include "global_structures.glsl"

SPEC_CONST_MAX_VECTORARROW_COUNT()

PUSH_CONSTANTS(InstancedVectorArrowParams)

layout(set = 0, binding = 0, std430) readonly buffer VectorArrowData {
    VectorArrow vectorArrowDatas[maxVectorArrowCount];
};
layout(set = 1, binding = 0, std430) readonly buffer VectorArrowIndices {
    uint vectorArrowIndexList_length;
    uint vectorArrowIndexList[maxVectorArrowCount];
};

// Per-vertex data
layout(location = 0) in vec2 v_pos;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec4 f_color;

void main() {
    const uint vectorIdx = vectorArrowIndexList[gl_InstanceIndex.x];
    VectorArrow vectorArrow = vectorArrowDatas[particleIdx];

    vec2 aspectRatio_scale = vec2(pConsts.render_heightDivWidth, 1);
    vec2 pos_01space = (v_pos.xy * vectorArrow.rotScale * pConsts.baseScale * aspectRatio_scale) + vectorArrow.pos;
    vec2 finalPos = (pos_01space * 2) - 1;
    gl_Position = vec4(finalPos.x, finalPos.y, 0, 1);
    f_uv = v_uv;
    // TODO - colorramp based on quantity?
    f_color = particle.color;
}