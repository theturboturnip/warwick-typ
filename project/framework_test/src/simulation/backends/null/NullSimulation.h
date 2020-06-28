//
// Created by samuel on 20/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/legacy.h"

/**
 * ISimulation that does not actually do any simulation. Used for testing legacy state stuff etc.
 */
class NullSimulation {
public:
    explicit NullSimulation(const LegacySimDump& dump);

    float tick();
    LegacySimDump dumpStateAsLegacy();

private:
    LegacySimDump m_state;
};
