//
// Created by samuel on 12/08/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"

#include <simulation/backends/cuda/utils/CudaUnified2DArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedRedBlackArray.cuh>

class CudaBackendV1 {
public:
    explicit CudaBackendV1(const SimSnapshot& s);
    ~CudaBackendV1();

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
    CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined> p;
    CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined> p_beta;
    CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined> rhs;
    CudaUnified2DArray<char> flag;
    CudaUnified2DArray<uint> fluidmask;
    CudaUnifiedRedBlackArray<uint, RedBlackStorage::RedBlackOnly> surroundmask;

    void dispatch_splitRedBlackCUDA(CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined>& to_split,
                                    dim3 blocksize_2d, dim3 gridsize_2d,
                                    CommonParams gpu_params);
    void dispatch_joinRedBlackCUDA(CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined>& to_join,
                                   dim3 blocksize_2d, dim3 gridsize_2d,
                                   CommonParams gpu_params);

    template<RedBlack Kind>
    void dispatch_poissonIterCUDA();

    cudaStream_t stream;
};