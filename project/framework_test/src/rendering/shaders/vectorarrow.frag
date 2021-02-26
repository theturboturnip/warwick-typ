#version 450

layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec4 f_color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(f_color.xyz, 1.0);
}
