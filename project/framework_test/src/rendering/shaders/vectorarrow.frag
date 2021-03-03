#version 450

#include "global_descriptor_sets.glsl"

PUSH_CONSTANTS(InstancedVectorArrowParams)

layout(location = 0) in vec2 f_uv;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(pConsts.color.rgb, 1.0);
}
