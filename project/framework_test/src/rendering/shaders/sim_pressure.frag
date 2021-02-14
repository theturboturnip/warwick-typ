#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D simBufferDataSampler;

void main() {
    // TODO - offset by 1/2 pixel
    vec2 offset = vec2(0,0);
    vec4 data = texture(simBufferDataSampler, uv + offset);

    outColor = data;
    return;

    if (data.w < 0.5) {
        // pixIdx is a valid fluid square, display pressure
        outColor = vec4(data.zzz, 1);
    } else {
        // pixIdx is a valid obstacle square, go green
        outColor = vec4(0, 1, 0, 1);
    }
}