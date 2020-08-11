//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"

class ISimTickedRunner {
protected:
    explicit ISimTickedRunner(float baseTimestep) : m_baseTimestep(baseTimestep) {}

    bool m_started = false;
    float m_currentTime = 0;
    const float m_baseTimestep;

public:
    virtual ~ISimTickedRunner() = default;

    [[nodiscard]] inline float started() const {
        return m_started;
    }

    [[nodiscard]] inline float currentTime() const {
        return m_currentTime;
    }

    virtual float tick() = 0;

    virtual void loadFromLegacy(const SimSnapshot& dump) = 0;
    virtual std::optional<SimSnapshot> get_snapshot() = 0;

    static std::unique_ptr<ISimTickedRunner> getForBackend(SimulationBackendEnum backendType, float baseTimestep);
};