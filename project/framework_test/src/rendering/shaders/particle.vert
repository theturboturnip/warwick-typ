#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_descriptor_sets.glsl"
#include "global_structures.glsl"

SPEC_CONST_MAX_PARTICLE_COUNT()

PUSH_CONSTANTS(InstancedParticleParams)

DS_PARTICLE_IMMUTABLE_INDEX_LIST(0, particlesToDrawIndexList)
DS_PARTICLE_INPUT_BUFFER(1, particleDatas)

// Per-vertex data
layout(location = 0) in vec2 v_pos;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec2 f_uv;

void main() {
    const uint particleIdx = particlesToDrawIndexList[gl_InstanceIndex.x];
    Particle particle = particleDatas[particleIdx];

    vec2 aspectRatio_scale = vec2(pConsts.render_heightDivWidth, 1);
    vec2 pos_01space = (v_pos.xy * pConsts.baseScale * aspectRatio_scale) + particlePos(particle.data);
    vec2 finalPos = (pos_01space * 2) - 1;
    gl_Position = vec4(finalPos.x, finalPos.y, 0, 1);
    f_uv = v_uv;
}