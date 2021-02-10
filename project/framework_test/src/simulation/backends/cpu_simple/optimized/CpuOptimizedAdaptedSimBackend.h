//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include <simulation/backends/cpu_simple/CpuOptimizedSimBackend.h>
#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "../CpuSimBackendBase.h"

class CpuOptimizedAdaptedSimBackend : public CpuOptimizedSimBackend {
public:
    explicit CpuOptimizedAdaptedSimBackend(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& s);

    float findMaxTimestep();
    int tick(float timestep);
};