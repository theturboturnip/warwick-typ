//
// Created by samuel on 22/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "CpuSimBackendBase.h"

class CpuSimpleSimBackend : public CpuSimBackendBase {
public:
    explicit CpuSimpleSimBackend(std::vector<BaseFrame> frames, const FluidParams& params, const SimSnapshot& s);

    float findMaxTimestep();
    void tick(float timestep, int frameToWriteIdx);

    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

    using Frame = BaseFrame;

private:
    std::vector<BaseFrame> frames;

    // This assumes only one frame exists, so we can copy the pointers
    float** u;
    float** v;
    float** f;
    float** g;
    float** p;
    float** rhs;
    char** flag;

    void computeTentativeVelocity(float del_t);

    void computeRhs(float del_t);

    int poissonSolver(float *res, int ifull);

    void updateVelocity(float del_t);

    void applyBoundaryConditions();
};