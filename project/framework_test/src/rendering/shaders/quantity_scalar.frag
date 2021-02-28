#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_descriptor_sets.glsl"
#include "global_structures.glsl"

SPEC_CONST_SCALAR_QUANTITY()

PUSH_CONSTANTS(QuantityScalarParams)

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

DS_SIM_DATA_SAMPLER(0, scalarQuantitySampler)
DS_GENERIC_INPUT_BUFFER(1, FloatRange, quantityDisplayRange)

vec4 convertQuantityToColor(float quantity, uint fallback) {
    // If outside the range, return the fallback
    if (quantity < quantityDisplayRange.min || quantity > quantityDisplayRange.max) {
        return unpackUnorm4x8(fallback);
    }
    // Get the quantity in the 0..1 range from quantityDisplayRange.min to quantityDisplayRange.max
    float quantityRel = (quantity - quantityDisplayRange.min) / (quantityDisplayRange.max - quantityDisplayRange.min);

    // Floating-point index - i.e. if 6.5, then halfway between color[6] and color[7]
    float index = quantityRel * 7;
    uint indexMin = uint(index); // Conversion to uint acts as floor(), which is what we want
    uint indexMax = min(7, indexMin + 1);
    // Fractional component = mix parameter
    // i.e. 6.5 => 0.5 betwen 6, 7
    float quantityRelBetweenIndices = index - float(indexMin);

    // Unpack the actual colors
    vec4 colorIndexMin = unpackUnorm4x8(pConsts.colorRange32Bit[indexMin]);
    vec4 colorIndexMax = unpackUnorm4x8(pConsts.colorRange32Bit[indexMax]);
    // Return the mix
    return mix(colorIndexMin, colorIndexMax, quantityRelBetweenIndices);
}

void main() {
    // TODO - offset by 1/2 pixel
    vec2 offset = vec2(0,0);
    vec4 data = texture(scalarQuantitySampler, uv + offset);
    float fluidmask = data.g;
    if (fluidmask >= 1.0) {
        // pixIdx is a valid fluid square, get the quantity
        if (scalarQuantity != ScalarQuantity_None) {
            float quantity = data.x;
            outColor = convertQuantityToColor(quantity, pConsts.fluidColor32Bit);
        } else {
            outColor = unpackUnorm4x8(pConsts.fluidColor32Bit);
        }
    } else {
        // pixIdx is a valid obstacle square
        outColor = unpackUnorm4x8(pConsts.obstacleColor32Bit);
    }

    outColor.w = 1;
}