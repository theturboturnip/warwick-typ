//
// Created by samuel on 12/08/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"

#include <simulation/backends/cuda/utils/CudaUnified2DArray.cuh>

class CudaBackendV1 {
public:
    explicit CudaBackendV1(const SimSnapshot& s);

    float findMaxTimestep();
    void tick(float timestep);

    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

private:
    const SimParams params;
    const Size<size_t> matrix_size;
    const Size<size_t> redblack_matrix_size;

    const int imax, jmax;
    const float del_x, del_y;
    const int ibound, ifluid;

    CudaUnified2DArray<float> u, v;
    CudaUnified2DArray<float> f, g;
    CudaUnified2DArray<float> p, p_red, p_black;
    CudaUnified2DArray<float> p_beta, p_beta_red, p_beta_black;
    CudaUnified2DArray<float> rhs, rhs_red, rhs_black;
    CudaUnified2DArray<char> flag;
    CudaUnified2DArray<int> fluidmask, surroundmask_red, surroundmask_black;
};