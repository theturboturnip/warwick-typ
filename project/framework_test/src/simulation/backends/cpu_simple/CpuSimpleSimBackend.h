//
// Created by samuel on 22/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/legacy.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "CpuSimBackendBase.h"

class CpuSimpleSimBackend : public CpuSimBackendBase {
public:
    explicit CpuSimpleSimBackend(const LegacySimDump& dump);

    float tick(float baseTimestep);

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