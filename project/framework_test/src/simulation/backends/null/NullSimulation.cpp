//
// Created by samuel on 20/06/2020.
//

#include "NullSimulation.h"

NullSimulation::NullSimulation(std::vector<Frame> frames, const FluidParams & params, const SimSnapshot& dump)
    : m_state(dump)
{}

LegacySimDump NullSimulation::dumpStateAsLegacy() {
    return m_state.to_legacy();
}

float NullSimulation::findMaxTimestep() {
    return -1;
}
void NullSimulation::tick(float timestep, int frameToWriteIdx) {}
SimSnapshot NullSimulation::get_snapshot() {
    return m_state;
}
