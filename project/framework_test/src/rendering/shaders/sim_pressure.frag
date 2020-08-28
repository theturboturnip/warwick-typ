#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WindowParams {
    uint pixelWidth;
    uint pixelHeight;
    uint columnStride;
    uint totalPixels;
};

layout(binding=0) buffer readonly velocity_x_buffer {
    float velocity_x[];
};
layout(binding=1) buffer readonly velocity_y_buffer {
    float velocity_y[];
};
layout(binding=2) buffer readonly pressure_buffer {
    float pressure[];
};
layout(binding=3) buffer readonly fluidmask_buffer {
    int fluidmask[];
};

void main() {
    uvec2 pixCoords = uvec2(uv * uvec2(pixelWidth, pixelHeight));
    uint pixIdx = (pixCoords.x * columnStride) + pixCoords.y;
    if (pixIdx >= totalPixels) {
        // Something's gone wrong - go purple!
        outColor = vec4(1, 0, 1, 0);
    } else if (fluidmask[pixIdx] != 0) {
        // pixIdx is a valid fluid square, go black for now TODO display pressure
        outColor = vec4(0, 0, 0, 1);
    } else {
        // pixIdx is a valid obstacle square, go green
        outColor = vec4(0, 1, 0, 1);
    }
}