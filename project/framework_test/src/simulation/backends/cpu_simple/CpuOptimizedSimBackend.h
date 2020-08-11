//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "CpuSimBackendBase.h"

class CpuOptimizedSimBackend : public CpuSimBackendBase {
public:
    explicit CpuOptimizedSimBackend(const SimSnapshot& s);

    float tick();

private:

    LegacyCompat2DBackingArray<float> p_beta, p_beta_red, p_beta_black;
    LegacyCompat2DBackingArray<float> p_red;
    LegacyCompat2DBackingArray<float> p_black;

    LegacyCompat2DBackingArray<float> rhs_red;
    LegacyCompat2DBackingArray<float> rhs_black;

    LegacyCompat2DBackingArray<int> fluidmask;
    LegacyCompat2DBackingArray<int> surroundmask_red;
    LegacyCompat2DBackingArray<int> surroundmask_black;

    void computeTentativeVelocity(float del_t);

    void computeRhs(float del_t);

    int poissonSolver(float *res, int ifull);

    void calculatePBeta();

    void splitToRedBlack(const LegacyCompat2DBackingArray<float>& joined, LegacyCompat2DBackingArray<float>& red, LegacyCompat2DBackingArray<float>& black);
    void joinRedBlack(LegacyCompat2DBackingArray<float>& joined, const LegacyCompat2DBackingArray<float>& red, const LegacyCompat2DBackingArray<float>& black);

    void updateVelocity(float del_t);

    void applyBoundaryConditions();

    void calculateFluidmask();
    void splitFluidmaskToSurroundedMask();
};