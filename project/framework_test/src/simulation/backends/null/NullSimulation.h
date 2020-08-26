//
// Created by samuel on 20/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"
#include <memory>
#include <simulation/file_format/FluidParams.h>
#include "simulation/memory/I2DAllocator.h"

/**
 * Simulation that does not actually do any simulation. Used for testing legacy state stuff etc.
 */
class NullSimulation {
public:
    explicit NullSimulation(I2DAllocator* alloc, const FluidParams & params, const SimSnapshot& dump);

    // findMaxTimestep() is called first, to determine the upper bound of the timestep.
    // return <0 if the runner can do anything
    float findMaxTimestep();
    // tick() will only ever be called with a timestep < findMaxTimestep()
    void tick(float timestep);

    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

private:
    SimSnapshot m_state;
};
