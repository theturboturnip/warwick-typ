//
// Created by samuel on 22/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "CpuSimBackendBase.h"

class CpuSimpleSimBackend : public CpuSimBackendBase {
public:
    explicit CpuSimpleSimBackend(I2DAllocator* alloc, const FluidParams & params, const SimSnapshot& s);

    float findMaxTimestep();
    void tick(float timestep);

private:
    /*template<typename T>
    std::vector<T*> make2DArrayFromBacking(const std::vector<T>& backing) {
        auto ptrArray = std::vector<T*>(m_state.totalElements(), nullptr);
        for (int i = 0; i < m_state.imax+2; i++) {

        }
    }*/

    void computeTentativeVelocity(float del_t);

    void computeRhs(float del_t);

    int poissonSolver(float *res, int ifull);

    void updateVelocity(float del_t);

    void applyBoundaryConditions();
};