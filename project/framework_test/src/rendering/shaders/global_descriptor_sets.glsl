#ifndef GLOBAL_DESCRIPTOR_SETS
#define GLOBAL_DESCRIPTOR_SETS

#include "global_structures.glsl"

#define ScalarQuantity_None 0
#define ScalarQuantity_VelocityX 1
#define ScalarQuantity_VelocityY 2
#define ScalarQuantity_VelocityMagnitude 3
#define ScalarQuantity_Pressure 4
#define ScalarQuantity_Vorticity 5

#define VectorQuantity_None 0
#define VectorQuantity_Velocity 1
#define VectorQuantity_VelocityX 2
#define VectorQuantity_VelocityY 3

#define SPEC_CONST_MAX_PARTICLE_COUNT() layout (constant_id = 0) const uint particleBufferLength = 1;
#define SPEC_CONST_MAX_PARTICLES_TO_EMIT_COUNT() layout (constant_id = 1) const uint particleToEmitBufferLength = 1;
#define SPEC_CONST_MAX_PARTICLE_EMITTER_COUNT() layout (constant_id = 2) const uint particleEmitterCount = 1;

#define SPEC_CONST_SCALAR_QUANTITY() layout (constant_id = 0) const uint scalarQuantity = ScalarQuantity_None;
#define SPEC_CONST_VECTOR_QUANTITY() layout (constant_id = 0) const uint vectorQuantity = VectorQuantity_None;

#define SPEC_CONST_MAX_VECTORARROW_COUNT()  layout (constant_id = 0) const uint maxVectorArrowCount = 1;


#define PUSH_CONSTANTS(TYPE) \
    layout(push_constant) uniform pushConstants { \
        TYPE pConsts; \
    };


#define DS_SIM_DATA_SAMPLER(SET, NAME) layout(set = SET, binding = 0) uniform sampler2D NAME;

#define DS_SIM_BUFFER_COPY_INPUT(SET) \
    layout (set=SET, binding=0) buffer readonly buf_u { \
        float velocity_x[]; \
    }; \
    layout (set=SET, binding=1) buffer readonly buf_v { \
        float velocity_y[]; \
    }; \
    layout (set=SET, binding=2) buffer readonly buf_p { \
        float pressure[]; \
    }; \
    layout (set=SET, binding=3) buffer readonly buf_fluidmask { \
        int fluidmask[]; \
    };
#define DS_SIM_BUFFER_COPY_OUTPUT(SET, NAME) \
    layout (set=SET, binding=0, rgba32f) uniform writeonly image2D NAME;

#define DS_IMAGE_INPUT(SET, FMT, NAME) \
    layout (set=SET, binding=0, FMT) uniform readonly image2D NAME;

#define DS_IMAGE_OUTPUT(SET, FMT, NAME) \
    layout (set=SET, binding=0, FMT) uniform writeonly image2D NAME;

#define DS_PARTICLE_INPUT_BUFFER(SET, NAME) \
    layout (set=SET, binding=0, std430) readonly buffer ParticleInBuffer { \
        Particle NAME[particleBufferLength]; \
    };

#define DS_PARTICLE_OUTPUT_BUFFER(SET, NAME) \
    layout (set=SET, binding=0, std430) buffer ParticleOutBuffer { \
        Particle NAME[particleBufferLength]; \
    };

#define DS_PARTICLE_MUTABLE_INDEX_LIST(SET, NAME) \
    layout (set=SET, binding=0, std430) buffer NAME##_Buffer { \
        coherent uint NAME##_length; \
        uint NAME[particleBufferLength]; \
    };

#define DS_PARTICLE_IMMUTABLE_INDEX_LIST(SET, NAME) \
    layout (set=SET, binding=0, std430) readonly buffer NAME##_Buffer { \
        coherent uint NAME##_length; \
        uint NAME[particleBufferLength]; \
    };

#define DS_PARTICLE_INDIRECT_CMDS(SET, NAME) \
    layout (set=SET, binding=0, std430) buffer ParticleIndirectCallsBuffer { \
        ParticleIndirectCommands NAME; \
    };

#define DS_PARTICLE_EMITTERS_DATA(SET, NAME) \
    layout (set=SET, binding=0, std430) readonly buffer ParticleEmittersBuffer { \
        ParticleEmitter NAME[particleEmitterCount]; \
    };

#define DS_PARTICLES_TO_EMIT_INPUT_DATA(SET, NAME) \
    layout (set=SET, binding=0, std430) readonly buffer ParticlesToEmitBuffer { \
        ParticleToEmitData NAME[particleToEmitBufferLength]; \
    };

#define DS_PARTICLES_TO_EMIT_OUTPUT_DATA(SET, NAME) \
    layout (set=SET, binding=0, std430) writeonly buffer ParticlesToEmitBuffer { \
        ParticleToEmitData NAME[particleToEmitBufferLength]; \
    };

#define DS_GENERIC_INPUT_BUFFER(SET, TYPE, NAME) \
    layout (set=SET, binding=0, std430) readonly buffer NAME##_Buffer { \
        TYPE NAME; \
    };

#define DS_GENERIC_OUTPUT_BUFFER(SET, TYPE, NAME) \
    layout (set=SET, binding=0, std430) readonly buffer NAME##_Buffer { \
        TYPE NAME; \
    };

// min/max float values used for unspecified numbers, i.e. when we're trying to reduce beyond the end of the buffer.
// From https://stackoverflow.com/a/47543127
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

#endif