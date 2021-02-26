#ifndef GLOBAL_STRUCTURES
#define GLOBAL_STRUCTURES

struct Particle {
// vec4(posX, posY, rot, none)
// pos in [0,1], rot in [0,2pi]?
    vec4 data;
// TODO other particle state
// Color?
    vec4 color;
};
#define particlePos(data) data.xy
#define particleRot(data) data.z

struct ParticleToEmitData {
    uint emitterIdx;
};

struct ParticleEmitter {
    vec4 position;
    vec4 color;
};

/**
* Push Constants
*/
struct SimDataBufferStats {
    uint sim_pixelWidth;
    uint sim_pixelHeight;
    uint sim_columnStride;
    uint sim_totalPixels;
};

struct ParticleSimulateParams {
    float timestep;
    float xLength, yLength;
};

struct InstancedParticleParams {
    float baseScale;
    float render_heightDivWidth;
};

struct ParticleKickoffParams {
    uint emitterCount;
};

struct QuantityScalarParams {
    // Create a color scale out of 8 32-bit color values.
    // colorRange[0] = minimum color, colorRange[7] = maximum color
    // Colors can be extracted with unpackUnorm4x8(), and packed with packSnorm4x8() (available in GLM core).
    uint colorRange32Bit[8];

    uint fluidColor32Bit;
    uint obstacleColor32Bit;
};

struct FloatRange {
    float min;
    float max;
};

struct ScalarExtractParams {
    uint simDataImage_width;
    uint simDataImage_height;
};

struct MinMaxReduceParams {
    uint bufferLength;
};

/**
* Indirect commands
*/
// Indirect compute command
struct VkDispatchIndirectCommand {
    uint    x;
    uint    y;
    uint    z;

    uint padding;
};

// Indirect indexed draw
struct VkDrawIndexedIndirectCommand {
    uint    indexCount;
    uint    instanceCount;
    uint    firstIndex;
    int     vertexOffset;
    uint    firstInstance;
};

struct VkDrawIndirectCommand {
    uint    vertexCount;
    uint    instanceCount;
    uint    firstVertex;
    uint    firstInstance;
};

struct ParticleIndirectCommands {
    // Add padding to the indirect dispatch commands so they have even length - RenderDoc doesn't like them otherwise

    VkDispatchIndirectCommand particleEmitCmd;

    VkDispatchIndirectCommand particleSimCmd;

    VkDrawIndirectCommand particleDrawCmd;

    uint particlesToEmitCount;
    uint particlesToSimCount;
};

#endif