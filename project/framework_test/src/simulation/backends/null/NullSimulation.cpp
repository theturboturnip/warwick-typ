//
// Created by samuel on 20/06/2020.
//

#include "NullSimulation.h"

void NullSimulation::loadFromLegacy(const LegacySimDump &dump) {
    this->m_started = true;
    this->m_state = dump;
}

LegacySimDump NullSimulation::dumpStateAsLegacy() {
    return m_state;
}

float NullSimulation::tick(float expectedTimestep) {
    if (!this->m_started)
        return 0;

    m_currentTime += expectedTimestep;
    return expectedTimestep;
}