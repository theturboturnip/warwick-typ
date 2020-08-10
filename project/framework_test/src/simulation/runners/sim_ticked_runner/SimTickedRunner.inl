//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/LegacySimDump.h"
#include "ISimTickedRunner.h"
#include "util/fatal_error.h"

/**
 *
 * @tparam SimBackend Backend for the simulation. Must have makeUniquePtrFromLegacy, dumpStateAsLegacy, tick
 */
template<typename SimBackend>
class SimTickedRunner : public ISimTickedRunner {
public:
    explicit SimTickedRunner(float baseTimestep) : ISimTickedRunner(baseTimestep) {}
    ~SimTickedRunner() override = default;

    void loadFromLegacy(const LegacySimDump& dump) override {
        m_started = true;

        m_backendData = std::make_unique<SimBackend>(dump, m_baseTimestep);
    }
    LegacySimDump dumpStateAsLegacy() override {
        if (!m_started)
            return LegacySimDump();

        return m_backendData->dumpStateAsLegacy();
    }

    float tick() override {
        DASSERT_M(m_backendData, "Backend data was null, make sure to load state into the simulation!");

        // TODO: timestep should be disregarded?
        float timestep = m_backendData->tick();
        m_currentTime += timestep;

        return timestep;
    }

private:
    std::unique_ptr<SimBackend> m_backendData = nullptr;
};