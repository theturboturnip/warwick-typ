#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_structures.glsl"

layout (constant_id = 0) const uint particleBufferLength = 0;

layout(push_constant) uniform pushConstants {
    InstancedParticleParams params;
};

layout(std140, set = 0, binding = 0) uniform ParticleBuffer {
    Particle particles[particleBufferLength];
};

in vec4 v_pos;
in vec2 v_uv;

out vec2 f_uv;

void main() {
    const Particle particleData = particles[gl_InstanceIndex];
    if (particleEnabled(particleData)) {
        vec2 finalPos = (v_pos * params.baseScale) + particlePos(particleData);
        gl_Position = vec4(finalPos.x, finalPos.y, 0, 1);
    } else {
        gl_Position = vec4(-5, -5, 0, 0);
    }
    f_uv = v_uv;
}