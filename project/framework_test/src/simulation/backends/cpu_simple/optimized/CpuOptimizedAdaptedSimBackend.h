//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "../CpuSimBackendBase.h"

class CpuOptimizedAdaptedSimBackend : public CpuSimBackendBase {
public:
    explicit CpuOptimizedAdaptedSimBackend(const FluidParams & params, const SimSnapshot& s);

    float findMaxTimestep();
    void tick(float timestep);

private:

    LegacyCompat2DBackingArray<float> p_beta, p_beta_red, p_beta_black;
    LegacyCompat2DBackingArray<float> p_red;
    LegacyCompat2DBackingArray<float> p_black;

    LegacyCompat2DBackingArray<float> rhs_red;
    LegacyCompat2DBackingArray<float> rhs_black;

    LegacyCompat2DBackingArray<int> fluidmask;
    LegacyCompat2DBackingArray<int> surroundmask_red;
    LegacyCompat2DBackingArray<int> surroundmask_black;
};