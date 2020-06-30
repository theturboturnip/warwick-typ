//
// Created by samuel on 30/06/2020.
//

#pragma once

#include <memory>
#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/LegacySimDump.h"

class ISim10sRunner {
protected:
    explicit ISim10sRunner() = default;

public:
    virtual ~ISim10sRunner() = default;
    virtual LegacySimDump runFor10s(const LegacySimDump& start, float baseTimestep) = 0;

    static std::unique_ptr<ISim10sRunner> getForBackend(SimulationBackendEnum backendType);
};