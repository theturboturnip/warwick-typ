
struct Particle {
// vec4(posX, posY, rot, enabled)
// pos in [0,1], rot in [0,2pi]?, enabled = 0 if not enabled, otherwise anything
    vec4 data;
// TODO other particle state
// Color?
    vec4 color;
};
#define particlePos(data) data.xy
#define particleRot(data) data.z
#define particleEnabled(data) (data.w != 0)

struct SimDataBufferStats {
    uint sim_pixelWidth;
    uint sim_pixelHeight;
    uint sim_columnStride;
    uint sim_totalPixels;
};

struct ParticleStepParams {
    float timestep;
    float xLength, yLength;
};

struct InstancedParticleParams {
    float baseScale;
};