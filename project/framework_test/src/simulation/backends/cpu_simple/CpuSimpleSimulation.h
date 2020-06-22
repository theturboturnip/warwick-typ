//
// Created by samuel on 22/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/legacy.h"
#include "simulation/interface.h"
#include "CpuSimpleSimulationBackend.h"

/**
 * Implementation of ISimulation based on the original ACA coursework.
 * Operates on the data inside the LegacySimDump
 */
class CpuSimpleSimulation : public ISimulation {
public:
    CpuSimpleSimulation() = default;
    ~CpuSimpleSimulation() override = default;

    void loadFromLegacy(const LegacySimDump& dump) override;
    LegacySimDump dumpStateAsLegacy() override;

    void tick(float expectedTimestep) override;

private:
    std::unique_ptr<CpuSimpleSimulationBackend> backendData = nullptr;

    //float findNextTimestepInterval
};
