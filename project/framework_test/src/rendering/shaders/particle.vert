#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_descriptor_sets.glsl"
#include "global_structures.glsl"

SPEC_CONST_PARTICLE_COUNT()

PUSH_CONSTANTS(InstancedParticleParams)

DS_PARTICLE_INPUT_BUFFER(0, particles)

layout(location = 0) in vec4 v_pos;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec2 f_uv;

void main() {
    const Particle particle = particles[gl_InstanceIndex];
    if (particleEnabled(particle.data)) {
        vec2 finalPos = (v_pos.xy * pConsts.baseScale) + particlePos(particle.data);
        gl_Position = vec4(finalPos.x, finalPos.y, 0, 1);
    } else {
        gl_Position = vec4(-5, -5, 0, 0);
    }
    f_uv = v_uv;
}