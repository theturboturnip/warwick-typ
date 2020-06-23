//
// Created by samuel on 22/06/2020.
//

#include "CpuSimpleSimulation.h"

void CpuSimpleSimulation::loadFromLegacy(const LegacySimDump &dump) {
    this->m_started = true;

    backendData = std::make_unique<CpuSimpleSimulationBackend>(dump);
}

LegacySimDump CpuSimpleSimulation::dumpStateAsLegacy() {
    auto state = LegacySimDump();
    if (!m_started)
        return state;

    state.params = backendData->params;
    state.u = backendData->u.getBacking();
    state.v = backendData->v.getBacking();
    state.p = backendData->p.getBacking();
    state.flag = backendData->flag.getBacking();
    return state;
}

float CpuSimpleSimulation::tick(float expectedTimestep) {
    if (!this->m_started) return 0;

    float timestep = backendData->tick();
    m_currentTime += timestep;

    return timestep;
}