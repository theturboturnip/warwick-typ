//
// Created by samuel on 12/08/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"

#include <simulation/backends/cuda/utils/CudaUnified2DArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedRedBlackArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedReducer.cuh>
#include <simulation/file_format/FluidParams.h>
#include "simulation/memory/SimulationAllocs.h"

struct CommonParams;

template<bool UnifiedMemory>
class CudaBackendV1 {
public:
    explicit CudaBackendV1(SimulationAllocs allocs, const FluidParams& params, const SimSnapshot& s);
    ~CudaBackendV1();

    float findMaxTimestep();
    void tick(float timestep);

    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

private:
    const FluidParams params;
    const SimSize simSize;
    const Size<uint32_t> matrix_size;
    const Size<uint32_t> redblack_matrix_size;

    const uint32_t imax, jmax;
    const float x_length, y_length;
    const float del_x, del_y;
    const int ibound, ifluid;

    CudaUnified2DArray<float, UnifiedMemory> u, v;
    CudaUnified2DArray<float, UnifiedMemory> f, g;
    CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined> p;
    CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::RedBlackOnly> p_buffered;
    CudaUnified2DArray<float, UnifiedMemory> p_sum_squares;
    CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined> p_beta;
    CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined> rhs;
    CudaUnified2DArray<char, UnifiedMemory> flag;
    CudaUnified2DArray<uint, UnifiedMemory> fluidmask;
    CudaUnifiedRedBlackArray<uint, UnifiedMemory, RedBlackStorage::RedBlackOnly> surroundmask;

    CudaReducer<UnifiedMemory, 128> reducer_fullsize;

    void dispatch_splitRedBlackCUDA(CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined>& to_split,
                                    dim3 gridsize_2d, dim3 blocksize_2d,
                                    CommonParams gpu_params);
    void dispatch_joinRedBlackCUDA(CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined>& to_join,
                                   dim3 gridsize_2d, dim3 blocksize_2d,
                                   CommonParams gpu_params);

    template<RedBlack Kind>
    void dispatch_poissonRedBlackCUDA(dim3 gridsize_redblack, dim3 blocksize_redblack, int iter, CommonParams gpu_params);

    cudaStream_t stream;
};