//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/LegacySimDump.h"

class ISimTickedRunner {
protected:
    explicit ISimTickedRunner() = default;

    bool m_started = false;
    float m_currentTime = 0;

public:
    virtual ~ISimTickedRunner() = default;

    inline float started() const {
        return m_started;
    }

    inline float currentTime() const {
        return m_currentTime;
    }

    virtual float tick(float baseTimestep) = 0;

    virtual void loadFromLegacy(const LegacySimDump& dump) = 0;
    virtual LegacySimDump dumpStateAsLegacy() = 0;

    static std::unique_ptr<ISimTickedRunner> getForBackend(SimulationBackendEnum backendType);
};