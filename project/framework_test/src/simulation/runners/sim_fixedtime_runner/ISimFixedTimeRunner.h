//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"
#include <memory>
#include <simulation/file_format/FluidParams.h>

class ISimFixedTimeRunner {
protected:
    explicit ISimFixedTimeRunner() = default;

public:
    virtual ~ISimFixedTimeRunner() = default;
    virtual SimSnapshot runForTime(const FluidParams & simParams, const SimSnapshot& start, float timeToRun, float forcedMaxTimestep) = 0;

    static std::unique_ptr<ISimFixedTimeRunner> getForBackend(SimulationBackendEnum backendType);
};