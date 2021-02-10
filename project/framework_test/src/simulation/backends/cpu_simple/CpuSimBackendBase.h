//
// Created by samuel on 28/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/LegacySimSize.h"
#include "simulation/file_format/SimSnapshot.h"
#include "util/LegacyCompat2DBackingArray.h"
#include <simulation/file_format/FluidParams.h>
#include "memory/FrameAllocator.h"
#include "simulation/memory/SimulationAllocs.h"

class CpuSimBackendBase {
public:
    class BaseFrame {
    public:
        BaseFrame(FrameAllocator<MType::Cpu>& alloc,
                  Size<uint32_t> paddedSize);

        Sim2DArray<float, MType::Cpu> u, v, f, g, p, rhs;
        Sim2DArray<char, MType::Cpu> flag;
    };

protected:
    //CpuSimBackendBase(const LegacySimDump& dump, float baseTimestep);
    explicit CpuSimBackendBase(const FluidParams& params, const SimSnapshot& s);

    const FluidParams params;
    const SimSize simSize;
    // Copies of the simulation parameter data for the C model
    const int imax, jmax;
    const float xlength, ylength;
    // Other simulation parameters (TODO: Some of these will need to go into SimulationParameters)
    const float delx, dely;
    const int ibound;
    const float ui, vi;
    const float Re, tau;
    const int itermax;
    const float eps, omega, gamma;
    const float baseTimestep;

    uint32_t getRequiredTimestepSubdivision(float umax, float vmax) const;
};