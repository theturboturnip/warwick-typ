//
// Created by samuel on 20/06/2020.
//

#include "NullSimulation.h"

NullSimulation::NullSimulation(const SimSnapshot& dump)
    : m_state(dump),
      m_baseTimestep(1.0f/dump.params.timestep_divisor) {}

LegacySimDump NullSimulation::dumpStateAsLegacy() {
    return m_state.to_legacy();
}

float NullSimulation::tick() {
    return m_baseTimestep;
}
SimSnapshot NullSimulation::get_snapshot() {
    return m_state;
}
