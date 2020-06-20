//
// Created by samuel on 20/06/2020.
//

#pragma once

#include "simulation/file_format/legacy.h"
#include "simulation/interface.h"

/**
 * ISimulation that does not actually do any simulation. Used for testing legacy state stuff etc.
 */
class NullSimulation : public ISimulation {
public:
    NullSimulation() = default;
    ~NullSimulation() override = default;

    void loadFromLegacy(const LegacySimDump& dump) override;
    LegacySimDump dumpStateAsLegacy() override;

    void tick(float expectedTimestep) override;

private:
    LegacySimDump m_state;
};
