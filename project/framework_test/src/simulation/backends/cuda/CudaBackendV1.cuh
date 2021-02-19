//
// Created by samuel on 12/08/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"

#include <simulation/backends/cuda/utils/CudaUnified2DArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedRedBlackArray.cuh>
#include <simulation/backends/cuda/utils/CudaUnifiedReducer.cuh>
#include <simulation/backends/cuda/utils/CudaUniquePtr.cuh>
#include <simulation/file_format/FluidParams.h>

#include "BaseCudaBackend.cuh"

#include <type_traits>
#include <simulation/backends/cuda/utils/CudaGraphCapture.cuh>

struct CommonParams;

template<bool UnifiedMemoryForExport>
class CudaBackendV1 : public BaseCudaBackend {
public:
    float findMaxTimestep();
    void tick(float timestep, int frameToWriteIdx);
    void copyToFrame(int frameToWriteIdx);

    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

    class Frame {
    public:
        constexpr static MType ExportMemType = UnifiedMemoryForExport ? MType::Cuda : MType::VulkanCuda;
        explicit Frame(FrameAllocator<ExportMemType>& exportAlloc,
                       FrameAllocator<MType::Cuda>& cudaAlloc,
                       Size<uint32_t> paddedSize);

        // TODO - would really prefer we didn't hold a reference to this after the fact.
        // The FrameAllocator isn't guaranteed to exist in the same place after this.
        FrameAllocator<MType::Cuda>& cudaAllocator;

        CudaUnified2DArray<uint, UnifiedMemoryForExport> fluidmask;

        CudaUnified2DArray<float, true> f, g;
        CudaUnifiedRedBlackArray<float, true, RedBlackStorage::RedBlackOnly> p_redblack_buffered;
        CudaUnified2DArray<float, true> p_sum_squares;
        CudaUnifiedRedBlackArray<float, true, RedBlackStorage::WithJoined> p_beta;
        CudaUnifiedRedBlackArray<float, true, RedBlackStorage::WithJoined> rhs;
        CudaUnified2DArray<char, true> flag;
        CudaUnifiedRedBlackArray<uint, true, RedBlackStorage::RedBlackOnly> surroundmask;

        CudaUnified2DArray<float, UnifiedMemoryForExport> u, v;
        CudaUnifiedRedBlackArray<float, UnifiedMemoryForExport, RedBlackStorage::WithJoined> p;

        CudaReducer<128> reducer_fullsize;
    };

    explicit CudaBackendV1(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& s);

private:
    std::vector<Frame> frames;
    int lastWrittenFrame;

    void resetFrame(Frame& frame, const SimSnapshot& s);

    const FluidParams fluidParams;
    const SimSize simSize;
    const Size<uint32_t> matrix_size;
    const Size<uint32_t> redblack_matrix_size;

    const uint32_t imax, jmax;
    const float x_length, y_length;
    const float del_x, del_y;
    const int ibound, ifluid;

    const dim3 blocksize_2d, gridsize_2d;
    const dim3 blocksize_redblack, gridsize_redblack;
    const dim3 blocksize_vertical, gridsize_vertical;
    const dim3 blocksize_horizontal, gridsize_horizontal;

    std::vector<CudaGraphCapture> poissonGraphs;

    template<MType SplitMType>
    void dispatch_splitRedBlackCUDA(SimRedBlackArray<float, SplitMType, RedBlackStorage::WithJoined>& to_split,
                                    CommonParams gpu_params);
    template<MType JoinMemType>
    void dispatch_joinRedBlackCUDA(SimRedBlackArray<float, JoinMemType, RedBlackStorage::WithJoined>& to_join,
                                   CommonParams gpu_params);

    template<RedBlack Kind>
    void dispatch_poissonRedBlackCUDA(int iter, Frame& frame, CommonParams gpu_params);
};