#version 450

#include "global_descriptor_sets.glsl"

SPEC_CONST_SCALAR_QUANTITY()

PUSH_CONSTANTS(QuantityScalarParams)


vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0)
);

layout(location = 0) out vec2 uv;

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = (pos + 1) / 2;
}