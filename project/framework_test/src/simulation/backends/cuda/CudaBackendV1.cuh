//
// Created by samuel on 12/08/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"

#include "simulation/memory/SimulationAllocs.h"
#include <simulation/backends/cuda/utils/CudaUnified2DArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedRedBlackArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedReducer.cuh>
#include <simulation/file_format/FluidParams.h>
#include <simulation/memory/CudaUnified2DAllocator.cuh>

#include "BaseCudaBackend.cuh"

struct CommonParams;

template<bool UnifiedMemoryForExport>
class CudaBackendV1 : public BaseCudaBackend {
public:
    explicit CudaBackendV1(SimulationAllocs allocs, const FluidParams& params, const SimSnapshot& s);

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

    CudaUnified2DArray<float, UnifiedMemoryForExport> u, v;
    CudaUnifiedRedBlackArray<float, UnifiedMemoryForExport, RedBlackStorage::WithJoined> p;
    CudaUnified2DArray<uint, UnifiedMemoryForExport> fluidmask;

    CudaUnified2DArray<float, true> f, g;
    CudaUnifiedRedBlackArray<float, true, RedBlackStorage::RedBlackOnly> p_buffered;
    CudaUnified2DArray<float, true> p_sum_squares;
    CudaUnifiedRedBlackArray<float, true, RedBlackStorage::WithJoined> p_beta;
    CudaUnifiedRedBlackArray<float, true, RedBlackStorage::WithJoined> rhs;
    CudaUnified2DArray<char, true> flag;
    CudaUnifiedRedBlackArray<uint, true, RedBlackStorage::RedBlackOnly> surroundmask;

    CudaReducer<true, 128> reducer_fullsize;

    template<bool UnifiedMemory>
    void dispatch_splitRedBlackCUDA(CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined>& to_split,
                                    dim3 gridsize_2d, dim3 blocksize_2d,
                                    CommonParams gpu_params);
    template<bool UnifiedMemory>
    void dispatch_joinRedBlackCUDA(CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined>& to_join,
                                   dim3 gridsize_2d, dim3 blocksize_2d,
                                   CommonParams gpu_params);

    template<RedBlack Kind>
    void dispatch_poissonRedBlackCUDA(dim3 gridsize_redblack, dim3 blocksize_redblack, int iter, CommonParams gpu_params);
};