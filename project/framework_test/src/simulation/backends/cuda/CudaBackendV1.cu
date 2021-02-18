//
// Created by samuel on 12/08/2020.
//

#include "CudaBackendV1.cuh"
#include <simulation/backends/cuda/kernels/redblack.cuh>
#include <util/check_cuda_error.cuh>

#include "simulation/backends/cuda/kernels/simple.cuh"
#include "simulation/backends/original/simulation.h"

inline float host_min(float x, float y) {
    return (x<y) ? x : y;
}

inline float host_max(float x, float y) {
    return (x>y) ? x : y;
}

template<bool UnifiedMemoryForExport>
CudaBackendV1<UnifiedMemoryForExport>::CudaBackendV1(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& s)
    : BaseCudaBackend(),
      fluidParams(params),
      simSize(s.simSize),
      matrix_size(simSize.padded_pixel_size),
      redblack_matrix_size(matrix_size.x, matrix_size.y / 2),

      // TODO eliminate/rename imax/jmax
      imax(simSize.internal_pixel_size.x),
      jmax(simSize.internal_pixel_size.y),
      x_length(simSize.physical_size.x),
      y_length(simSize.physical_size.y),
      del_x(simSize.del_x()),
      del_y(simSize.del_y()),
      ibound(s.get_boundary_cell_count()),
      ifluid(imax * jmax - ibound),

      blocksize_2d(1, 64),
      gridsize_2d(
              (matrix_size.x + blocksize_2d.x - 1) / blocksize_2d.x,
              (matrix_size.y + blocksize_2d.y - 1) / blocksize_2d.y
      ),
      blocksize_redblack(1, 64),
      gridsize_redblack(
              (redblack_matrix_size.x + blocksize_redblack.x - 1) / blocksize_redblack.x,
              (redblack_matrix_size.y + blocksize_redblack.y - 1) / blocksize_redblack.y
      ),
      blocksize_vertical(32),
      gridsize_vertical((matrix_size.y + blocksize_vertical.x - 1) / blocksize_vertical.x),
      blocksize_horizontal(32),
      gridsize_horizontal((matrix_size.x + blocksize_horizontal.x - 1) / blocksize_horizontal.x),

      poissonGraph(stream),

      frames(std::move(frames)),
      lastWrittenFrame(0)
{
    DASSERT(!this->frames.empty());

    // Use this->frames because `frames` on it's own is the newly-removed argument
    for (auto& frame : this->frames) {
        resetFrame(frame, s);
    }
}

template<bool UnifiedMemoryForExport>
float CudaBackendV1<UnifiedMemoryForExport>::findMaxTimestep() {
    Frame& frameWithVelocity = frames[lastWrittenFrame];

    float delta_t = -1;
    auto fabsf_lambda = [] __device__ (float x) { return fabsf(x); };
    auto max_lambda = [] __device__ (float x, float y) { return max(x, y); };
    // TODO - having multiple reducers here would be more efficient - could dispatch both, and then wait for one then the other?
    float u_max = frameWithVelocity.reducer_fullsize.map_reduce(frameWithVelocity.u, fabsf_lambda, max_lambda, stream);
    u_max = host_max(u_max, 1.0e-10);
    float v_max = frameWithVelocity.reducer_fullsize.map_reduce(frameWithVelocity.v, fabsf_lambda, max_lambda, stream);
    v_max = host_max(v_max, 1.0e-10);

    float delt_u = del_x/u_max;
    float delt_v = del_y/v_max;
    // This used to be deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;
    // the original version has 2.0 at the end, but this only ends up doing the rest of the equation, promoting it to double, dividing it, and demoting back to int.
    // this is equivalent to dividing by 2.0f without any double-promotions.
    float deltRe = 1.0f / (1.0f/(del_x*del_x)+1.0f/(del_y*del_y)) * fluidParams.Re / 2.0f;

    if (delt_u<delt_v) {
        delta_t = host_min(delt_u, deltRe);
    } else {
        delta_t = host_min(delt_v, deltRe);
    }
    delta_t = fluidParams.timestep_safety * (delta_t); // multiply by safety factor

//    printf("GPU del_t\n");
//    printf("u_max: %a\tv_max: %a\n", u_max, v_max);
//    printf("delt_u: %a\tdelt_v: %a\tdelt_re: %a\n", delt_u, delt_v, deltRe);
//    printf("delta_t: %a\n", delta_t);

//    float cpu_delta_t = -1;
//    OriginalOptimized::setTimestepInterval(&cpu_delta_t,
//                        imax, jmax,
//                        del_x, del_y,
//                        u.as_cpu(), v.as_cpu(),
//                        params.Re,
//                        params.timestep_safety
//                        );

//    printf("CPU del_t\ndelta_t: %a\n", delta_t);

    DASSERT(delta_t != -1);
    return delta_t;
}

template<bool UnifiedMemoryForExport>
template<MType SplitMemType>
void CudaBackendV1<UnifiedMemoryForExport>::dispatch_splitRedBlackCUDA(
        SimRedBlackArray<float, SplitMemType, RedBlackStorage::WithJoined>& to_split,
        CommonParams params)
{
    static_assert(SplitMemType == MType::Cuda || SplitMemType == MType::VulkanCuda, "Only works on CUDA memory");

    split_redblack_simple<<<gridsize_2d, blocksize_2d, 0, stream>>>(
            to_split.joined.as_cuda(),
            to_split.red.as_cuda(), to_split.black.as_cuda(),
            params
    );
    CHECK_KERNEL_ERROR();
}
template<bool UnifiedMemoryForExport>
template<MType JoinMemType>
void CudaBackendV1<UnifiedMemoryForExport>::dispatch_joinRedBlackCUDA(
        SimRedBlackArray<float, JoinMemType, RedBlackStorage::WithJoined>& to_join,
        CommonParams params)
{
    static_assert(JoinMemType == MType::Cuda || JoinMemType == MType::VulkanCuda, "Only works on CUDA memory");

    join_redblack_simple<<<gridsize_2d, blocksize_2d,0, stream>>>(
            to_join.red.as_cuda(), to_join.black.as_cuda(),
            to_join.joined.as_cuda(),
            params
    );
    CHECK_KERNEL_ERROR();
}

template<bool UnifiedMemoryForExport>
template<RedBlack Kind>
void CudaBackendV1<UnifiedMemoryForExport>::dispatch_poissonRedBlackCUDA(int iter, Frame& frame, CommonParams gpu_params) {
    // For a p_red computation: do p_red/p_buffered_black into p_buffered_red, while copying p_buffered_black into p_black.
    // Modern Nvidia GPUs can do parallel memcpy and compute, so this shouldn't take longer

    // Note - all p.get<Kind> functions need to have the template specifier: "p.get(Kind)".
    // This is because:
    //  1. CudaBackendV1 is templated on UnifiedMemory
    //  2. Which causes the types of p, p_buffered, rhs etc. to be *dependent* on the value of UnifiedMemory.
    //  3. Because of template specializations, the C++ compiler can't know ahead of time if p.get will be a template function or a data member.
    //  4. This means the C++ compiler can't tell what p.get(Kind) means - is it (p.get) < Kind > ()? or is it a template function invocation?
    //  5. Using the '.template' syntax tells the C++ compiler to explicitly expect a template function here. It's ugly as hell, but it works.
    // The way to fix this is to stop using shitty languages like C++. However, given the state of CUDA on better languages like Rust is kinda poor,
    // this isn't feasible right now.

    constexpr bool DoubleBuffer = false;

    if (DoubleBuffer) {
        poisson_single_tick<<<gridsize_redblack, blocksize_redblack, 0, stream>>>(
                frame.p.get(Kind).as_cuda(),
                frame.p_redblack_buffered.get_other(Kind).as_cuda(),
                frame.rhs.get(Kind).as_cuda(),
                frame.p_beta.get(Kind).as_cuda(),

                frame.p_redblack_buffered.get(Kind).as_cuda(),

                (Kind == RedBlack::Black) ? 1 : 0,// 0 if red, 1 if black

                fluidParams.poisson_omega,

                iter,

                gpu_params);

        // TODO - this needs to be done in a separate stream to overlap
        frame.p.get_other(Kind).dispatch_memcpy_in(frame.p_redblack_buffered.get_other(Kind), stream);
    } else {
        poisson_single_tick<<<gridsize_redblack, blocksize_redblack, 0, stream>>>(
                frame.p.get(Kind).as_cuda(),
                frame.p.get_other(Kind).as_cuda(),
                frame.rhs.get(Kind).as_cuda(),
                frame.p_beta.get(Kind).as_cuda(),

                frame.p.get(Kind).as_cuda(),

                (Kind == RedBlack::Black) ? 1 : 0,// 0 if red, 1 if black

                fluidParams.poisson_omega,

                iter,

                gpu_params);
    }
    CHECK_KERNEL_ERROR();
}

template<bool UnifiedMemoryForExport>
LegacySimDump CudaBackendV1<UnifiedMemoryForExport>::dumpStateAsLegacy() {
    const Frame& frameToDump = frames[lastWrittenFrame];

    CHECKED_CUDA(cudaStreamSynchronize(stream));
    auto dump = LegacySimDump(simSize.to_legacy());
    dump.u = frameToDump.u.extract_data();
    dump.v = frameToDump.v.extract_data();
    dump.p = frameToDump.p.joined.extract_data();
    dump.flag = frameToDump.flag.extract_data();
    return dump;
}
template<bool UnifiedMemoryForExport>
SimSnapshot CudaBackendV1<UnifiedMemoryForExport>::get_snapshot() {
    return SimSnapshot::from_legacy(dumpStateAsLegacy());
}

template<bool UnifiedMemoryForExport>
void CudaBackendV1<UnifiedMemoryForExport>::resetFrame(CudaBackendV1::Frame &frame, const SimSnapshot &s) {
    auto gpu_params = CommonParams{
            .size = uint2{matrix_size.x, matrix_size.y},
            .redblack_size = uint2{redblack_matrix_size.x, redblack_matrix_size.y},
            .col_pitch_4byte=frame.u.stats.col_pitch,
            .col_pitch_redblack=frame.rhs.red.stats.col_pitch,
            .deltas = float2{del_x, del_y},
    };

    frame.u.memcpy_in(s.velocity_x);
    frame.v.memcpy_in(s.velocity_y);
    frame.fluidmask.memcpy_in(s.get_fluidmask());

    // Setup frame.p, frame.p_redblack_buffered
    {
        // Copy data into p.joined, then split it into
        frame.p.joined.memcpy_in(s.pressure);

        // split p.joined into p.red, p.black
        dispatch_splitRedBlackCUDA(frame.p, gpu_params);
        cudaStreamSynchronize(stream);

        // copy p.red, p.black into p_redblack_buffered
        frame.p_redblack_buffered.red.memcpy_in(frame.p.red);
        frame.p_redblack_buffered.black.memcpy_in(frame.p.black);
    }

    frame.flag.memcpy_in(s.get_legacy_cell_flags());

    frame.rhs.zero_out();

    frame.f.zero_out();
    frame.g.zero_out();


    // TODO - remove poisson_error_threshold from args
    OriginalOptimized::calculatePBeta(frame.p_beta.joined.as_cpu(), frame.flag.as_cpu(),
                                      imax, jmax, del_x, del_y,
                                      fluidParams.poisson_error_threshold, fluidParams.poisson_omega);
//    OriginalOptimized::splitToRedBlack(frame.p_beta.joined.as_cpu(),
//                                       frame.p_beta.red.as_cpu(), frame.p_beta.black.as_cpu(),
//                                       imax, jmax);
    dispatch_splitRedBlackCUDA(frame.p_beta, gpu_params);
    cudaStreamSynchronize(stream);

    // Calculate the fluidmask and surroundedmask items
    // This currently doesn't have a CUDA kernel.
    if constexpr (UnifiedMemoryForExport) {
        OriginalOptimized::splitFluidmaskToSurroundedMask((const int **) (frame.fluidmask.as_cpu()),
                                                          (int **) frame.surroundmask.red.as_cpu(), (int **) frame.surroundmask.black.as_cpu(),
                                                          imax, jmax);
    } else {
        CudaUnified2DArray<uint32_t, true> fluidmask_unified(frame.cudaAllocator, matrix_size);
        fluidmask_unified.memcpy_in(frame.fluidmask);
        OriginalOptimized::splitFluidmaskToSurroundedMask((const int **) (fluidmask_unified.as_cpu()),
                                                          (int **) frame.surroundmask.red.as_cpu(), (int **) frame.surroundmask.black.as_cpu(),
                                                          imax, jmax);
    }

    cudaStreamSynchronize(stream);
    CHECK_KERNEL_ERROR();
}

template<bool UnifiedMemoryForExport>
void CudaBackendV1<UnifiedMemoryForExport>::tick(float timestep, int frameToWriteIdx) {
    const auto& previousFrame = frames[lastWrittenFrame];
    auto& frame = frames[frameToWriteIdx];

    auto gpu_params = CommonParams{
            .size = uint2{matrix_size.x, matrix_size.y},
            .redblack_size = uint2{redblack_matrix_size.x, redblack_matrix_size.y},
            .col_pitch_4byte=frame.u.stats.col_pitch,
            .col_pitch_redblack=frame.rhs.red.stats.col_pitch,
            .deltas = float2{del_x, del_y},
    };

    // Compute Tentative Velocity from previousFrame.u,v => frame.f,g
    {
        computeTentativeVelocity_apply<<<gridsize_2d, blocksize_2d, 0, stream>>>(
                previousFrame.u.as_cuda(), previousFrame.v.as_cuda(), frame.fluidmask.as_cuda(),
                frame.f.as_cuda(), frame.g.as_cuda(),
                gpu_params, timestep, fluidParams.gamma, fluidParams.Re
        );

        computeTentativeVelocity_postproc_vertical<<<gridsize_vertical, blocksize_vertical, 0, stream>>>(
                previousFrame.u.as_cuda(), frame.f.as_cuda(), gpu_params);
        computeTentativeVelocity_postproc_horizontal<<<gridsize_horizontal, blocksize_horizontal, 0, stream>>>(
                previousFrame.v.as_cuda(), frame.g.as_cuda(), gpu_params);

        CHECK_KERNEL_ERROR();
    }

    // Compute RHS from frame.f,g => frame.rhs
    {
        computeRHS_1per<<<gridsize_2d, blocksize_2d, 0, stream>>>(frame.f.as_cuda(), frame.g.as_cuda(), frame.fluidmask.as_cuda(),
                                                                  frame.rhs.joined.as_cuda(), gpu_params, timestep);
        // Split RHS
        dispatch_splitRedBlackCUDA(frame.rhs, gpu_params);
        CHECK_KERNEL_ERROR();
    }

    // Compute poisson from previousFrame.p, frame.p_beta,rhs,fluidmask,surroundmask,flag
    {
        // TODO - split previousFrame.p => frame.p.red, frame.p.black
        //  this could be done directly? but is likely more trouble than it's worth.
        // TODO - if frame == previousFrame this isn't needed at all.
        frame.p.joined.dispatch_memcpy_in(previousFrame.p.joined, stream);
        dispatch_splitRedBlackCUDA(frame.p, gpu_params);

        if (ifluid > 0) {
            constexpr bool UseCPUPoisson = false;
            if constexpr (UnifiedMemoryForExport && UseCPUPoisson) {
                OriginalOptimized::poissonSolver<false>(frame.p.joined.as_cpu(), frame.p.red.as_cpu(), frame.p.black.as_cpu(),
                                                        frame.p_beta.joined.as_cpu(), frame.p_beta.red.as_cpu(), frame.p_beta.black.as_cpu(),
                                                        frame.rhs.joined.as_cpu(), frame.rhs.red.as_cpu(), frame.rhs.black.as_cpu(),
                                                        (int **) frame.fluidmask.as_cpu(), (int **) frame.surroundmask.black.as_cpu(),
                                                        frame.flag.as_cpu(), imax, jmax,
                                                        del_x, del_y,
                                                        fluidParams.poisson_error_threshold, fluidParams.poisson_max_iterations, fluidParams.poisson_omega,
                                                        ifluid);
            } else {
                if (!poissonGraph.recorded) {
                    poissonGraph.record([&, this]() {
                        // Sum of squares of pressure - reduction
                        // poisson_pSquareSumReduce(p.joined.as_cuda(), p_sum_squares.as_cuda())
                        // p0 = p_sum_squares.as_cpu(?????)???
                        // TODO - accessing memory like this is very convenient with managed memory
                        //  We *might* be able to us VK_EXT_external_memory_host to import CUDA Managed Memory as Vulkan, bypassing Vulkan allocations

                        //const float partial_res_sqr_thresh = params.poisson_error_threshold * p0 * params.poisson_error_threshold * p0 * (float)ifluid;

                        // Red/Black SOR-iteration
                        for (int iter = 0; iter < fluidParams.poisson_max_iterations; iter++) {
                            //  redblack<Red>();
                            dispatch_poissonRedBlackCUDA<RedBlack::Red>(iter, frame, gpu_params);
                            //  float approxRes = redblack<Black>(); (capture approximate residual here)
                            //float approxRes; // TODO - ???
                            dispatch_poissonRedBlackCUDA<RedBlack::Black>(iter, frame, gpu_params);
                            //  [ IMPLICIT STREAM SYNC FOR RESIDUAL ]
                            // [ NOT NECESSARY WHEN NOT CALCULATING RESIDUAL ]
                            //  if (approxRes < partial_res_sqr_thresh)
                            //      TODO - necessary to capture full res at all? if the approxRes is actually accurate, then maybe not
                            //       If we have to calculate this we may have to merge pressure here
                            //      break;
                            //  TODO - dynamic error
                        }

                        // join p
                        dispatch_joinRedBlackCUDA(frame.p, gpu_params);
                        // Stream sync not necessary here, because the rest is CUDA
                    });
                }
                poissonGraph.execute();
            }
        }
    }

    // Update Velocity from frame.f,g,p,fluidmask => frame.u,v
    {
        //    OriginalOptimized::updateVelocity(u.as_cpu(), v.as_cpu(),
//                       f.as_cpu(), g.as_cpu(),
//                       p.as_cpu(), flag.as_cpu(),
//                       imax, jmax, timestep, del_x, del_y);
        updateVelocity_1per<<<gridsize_2d, blocksize_2d, 0, stream>>>(frame.f.as_cuda(), frame.g.as_cuda(), frame.p.joined.as_cuda(), frame.fluidmask.as_cuda(),
                                                                      frame.u.as_cuda(), frame.v.as_cuda(),
                                                                      gpu_params, timestep);
        CHECK_KERNEL_ERROR();
    }

    // Compute boundary conditions from frame.u,v => frame.u,v
    {
        boundaryConditions_preproc_vertical<<<gridsize_vertical, blocksize_vertical, 0, stream>>>(frame.u.as_cuda(),
                                                                                                  frame.v.as_cuda(),
                                                                                                  gpu_params);
        boundaryConditions_preproc_horizontal<<<gridsize_horizontal, blocksize_horizontal, 0, stream>>>(
                frame.u.as_cuda(), frame.v.as_cuda(), gpu_params);

        boundaryConditions_apply<<<gridsize_2d, blocksize_2d, 0, stream>>>(frame.flag.as_cuda(),
                                                                           frame.u.as_cuda(), frame.v.as_cuda(),
                                                                           gpu_params);

        boundaryConditions_inputflow_west_vertical<<<gridsize_vertical, blocksize_vertical, 0, stream>>>(
                frame.u.as_cuda(), frame.v.as_cuda(),
                float2{fluidParams.initial_velocity_x, fluidParams.initial_velocity_y},
                gpu_params
        );
        CHECK_KERNEL_ERROR();

        //    OriginalOptimized::applyBoundaryConditions(u2.as_cpu(), v2.as_cpu(), flag.as_cpu(), imax, jmax, params.initial_velocity_x, params.initial_velocity_y);
    }

    // Done!
    lastWrittenFrame = frameToWriteIdx;
}

template<bool UnifiedMemoryForExport>
void CudaBackendV1<UnifiedMemoryForExport>::copyToFrame(int frameToWriteIdx) {
    const Frame& copyFrom = frames[lastWrittenFrame];
    Frame& copyTo = frames[frameToWriteIdx];

    // Values of f,g,rhs etc. aren't persisted between invocations, so we don't need to copy them
    copyTo.u.dispatch_memcpy_in(copyFrom.u, stream);
    copyTo.v.dispatch_memcpy_in(copyFrom.v, stream);
    copyTo.p.joined.dispatch_memcpy_in(copyFrom.p.joined, stream);
    copyTo.p.red.dispatch_memcpy_in(copyFrom.p.red, stream);
    copyTo.p.black.dispatch_memcpy_in(copyFrom.p.black, stream);

    lastWrittenFrame = frameToWriteIdx;
}

template<bool UnifiedMemoryForExport>
CudaBackendV1<UnifiedMemoryForExport>::Frame::Frame(FrameAllocator<ExportMemType>& exportAlloc,
                                                    FrameAllocator<MType::Cuda>& cudaAlloc,
                                                    Size<uint32_t> paddedSize)
    : cudaAllocator(cudaAlloc),

      // Exportable (i.e. renderable) matrices
      u(exportAlloc, paddedSize),
      v(exportAlloc, paddedSize),
      p(exportAlloc, paddedSize),
      fluidmask(exportAlloc, paddedSize),

      // Other matrices can be cuda-only.
      f(cudaAlloc, paddedSize),
      g(cudaAlloc, paddedSize),
      p_redblack_buffered(cudaAlloc, paddedSize),
      p_sum_squares(cudaAlloc, paddedSize),
      p_beta(cudaAlloc, paddedSize),
      rhs(cudaAlloc, paddedSize),
      flag(cudaAlloc, paddedSize),
      surroundmask(cudaAlloc, paddedSize),

      reducer_fullsize(cudaAlloc, paddedSize.area())
{}

template class CudaBackendV1<true>;
template class CudaBackendV1<false>;