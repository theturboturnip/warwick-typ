//
// Created by samuel on 20/06/2020.
//

#include "NullSimulation.h"

NullSimulation::NullSimulation(const LegacySimDump& dump) : m_state(dump) {}

LegacySimDump NullSimulation::dumpStateAsLegacy() {
    return m_state;
}

float NullSimulation::tick() {
    // TODO: This is bad behaviour, we should establish a base timestep
    return 1;
}

std::unique_ptr<NullSimulation> NullSimulation::makeUniquePtrFromLegacy(const LegacySimDump& dump) {
    return std::make_unique<NullSimulation>(dump);
}
