//
// Created by samuel on 28/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/LegacySimulationParameters.h"
#include "simulation/file_format/SimSnapshot.h"
#include "util/LegacyCompat2DBackingArray.h"
#include <simulation/file_format/FluidParams.h>

class CpuSimBackendBase {
public:
    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

protected:
    //CpuSimBackendBase(const LegacySimDump& dump, float baseTimestep);
    explicit CpuSimBackendBase(const FluidParams & params, const SimSnapshot& s);

    const FluidParams params;
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

    LegacyCompat2DBackingArray<float> u, v, f, g, p, rhs;
    LegacyCompat2DBackingArray<char> flag;

    uint32_t getRequiredTimestepSubdivision(float umax, float vmax) const;
};