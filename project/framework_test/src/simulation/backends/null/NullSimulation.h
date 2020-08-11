//
// Created by samuel on 20/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include <memory>
#include "simulation/file_format/SimSnapshot.h"

/**
 * ISimulation that does not actually do any simulation. Used for testing legacy state stuff etc.
 */
class NullSimulation {
public:
    explicit NullSimulation(const SimSnapshot& dump);

    float tick();
    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

private:
    SimSnapshot m_state;
    const float m_baseTimestep;
};
