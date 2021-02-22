#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_descriptor_sets.glsl"
#include "global_structures.glsl"

PUSH_CONSTANTS(InstancedParticleParams)

// Per-vertex data
layout(location = 0) in vec2 v_pos;
layout(location = 1) in vec2 v_uv;

// Per-instance data
layout(location = 2) in vec4 particle_data;
layout(location = 3) in vec4 particle_color;

layout(location = 0) out vec2 f_uv;

void main() {
    if (particleEnabled(particle_data)) {
        vec2 finalPos = (v_pos.xy * pConsts.baseScale) + particlePos(particle_data);
        gl_Position = vec4(finalPos.x, finalPos.y, 0, 1);
    } else {
        gl_Position = vec4(-5, -5, 0, 0);
    }
    f_uv = v_uv;
}