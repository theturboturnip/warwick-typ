//
// Created by samuel on 20/06/2020.
//

#pragma once

#include <memory>

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"
#include "simulation/file_format/FluidParams.h"
#include "simulation/memory/SimulationAllocs.h"
#include "memory/FrameAllocator.h"

/**
 * Simulation that does not actually do any simulation. Used for testing legacy state stuff etc.
 */
class NullSimulation {
public:
    // findMaxTimestep() is called first, to determine the upper bound of the timestep.
    // return <0 if the runner can do anything
    float findMaxTimestep();
    // tick() will only ever be called with a timestep < findMaxTimestep()
    void tick(float timestep, int frameToWriteIdx);

    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

    class Frame {
    public:
        Frame(FrameAllocator<MType::Cpu>& alloc, Size<uint32_t> paddedSize){}
    };

    explicit NullSimulation(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& dump);
private:
    SimSnapshot m_state;
};
