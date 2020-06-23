//
// Created by samuel on 22/06/2020.
//

#pragma once

#include "simulation/file_format/legacy.h"
#include "util/LegacyCompat2DBackingArray.h"

// TODO: Rename, CpuSimpleSimulation is already a backend itself
class CpuSimpleSimulationBackend {
public:
    explicit CpuSimpleSimulationBackend(const LegacySimDump& dump);

    float tick();

    SimulationParameters params;
    // Copies of the simulation parameter data for the C model
    const int imax, jmax;
    const float xlength, ylength;
    // Other simulation parameters (TODO: Some of these will need to go into SimulationParameters)
    const float delx, dely;
    const int ibound;
    const float ui, vi;
    const float Re, tau;
    const int itermax;
    const float eps, omega, gamma;
    float del_t; // Not const, this can change over times

    LegacyCompat2DBackingArray<float> u, v, f, g, p, rhs;
    LegacyCompat2DBackingArray<char> flag;

private:
    /*template<typename T>
    std::vector<T*> make2DArrayFromBacking(const std::vector<T>& backing) {
        auto ptrArray = std::vector<T*>(m_state.totalElements(), nullptr);
        for (int i = 0; i < m_state.imax+2; i++) {

        }
    }*/

    void computeTentativeVelocity();

    void computeRhs();

    int poissonSolver(float *res, int ifull);

    void updateVelocity();

    void setTimestepInterval();

    void applyBoundaryConditions();
};