#define SPEC_CONST_PARTICLE_COUNT() layout (constant_id = 0) const uint particleBufferLength = 1;


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
#define DS_SIM_BUFFER_COPY_OUTPUT(SET) \
    layout (set=SET, binding=0, rgba32f) uniform writeonly image2D resultImage;

#define DS_PARTICLE_INPUT_BUFFER(SET, NAME) \
    layout (set=SET, binding=0, std140) readonly buffer ParticleInBuffer { \
        Particle NAME[particleBufferLength]; \
    };

#define DS_PARTICLE_OUTPUT_BUFFER(SET, NAME) \
    layout (set=SET, binding=0, std140) buffer ParticleOutBuffer { \
        Particle NAME[particleBufferLength]; \
    };