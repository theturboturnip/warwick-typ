//
// Created by samuel on 30/06/2020.
//

#pragma once

#include <memory>
#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/LegacySimDump.h"

class ISimFixedTimeRunner {
protected:
    explicit ISimFixedTimeRunner() = default;

public:
    virtual ~ISimFixedTimeRunner() = default;
    virtual LegacySimDump runForTime(const LegacySimDump& start, float baseTimestep, float timeToRun) = 0;

    static std::unique_ptr<ISimFixedTimeRunner> getForBackend(SimulationBackendEnum backendType);
};