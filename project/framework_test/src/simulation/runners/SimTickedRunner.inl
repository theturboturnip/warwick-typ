//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/legacy.h"
#include "ISimTickedRunner.h"
#include "util/fatal_error.h"

/**
 *
 * @tparam SimBackend Backend for the simulation. Must have makeUniquePtrFromLegacy, dumpStateAsLegacy, tick
 */
template<typename SimBackend>
class SimTickedRunner : public ISimTickedRunner {
public:
    SimTickedRunner() = default;
    ~SimTickedRunner() override = default;

    void loadFromLegacy(const LegacySimDump& dump) override {
        m_started = true;

        m_backendData = std::make_unique<SimBackend>(dump);
    }
    LegacySimDump dumpStateAsLegacy() override {
        if (!m_started)
            return LegacySimDump();

        return m_backendData->dumpStateAsLegacy();
    }

    float tick(float baseTimestep) override {
        DASSERT_M(m_backendData, "Backend data was null, make sure to load state into the simulation!");

        float timestep = m_backendData->tick(baseTimestep);
        m_currentTime += timestep;

        return timestep;
    }

private:
    std::unique_ptr<SimBackend> m_backendData = nullptr;
};