//
// Created by samuel on 20/06/2020.
//

#include "NullSimulation.h"

NullSimulation::NullSimulation(const LegacySimDump& dump, float baseTimestep) : m_state(dump), m_baseTimestep(baseTimestep) {}

LegacySimDump NullSimulation::dumpStateAsLegacy() {
    return m_state;
}

float NullSimulation::tick() {
    return m_baseTimestep;
}
