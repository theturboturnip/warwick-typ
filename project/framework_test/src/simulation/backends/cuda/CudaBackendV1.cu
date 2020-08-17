//
// Created by samuel on 12/08/2020.
//

#include "CudaBackendV1.cuh"
#include <simulation/backends/cuda/kernels/redblack.cuh>

#include "simulation/backends/original/simulation.h"
#include "simulation/backends/cuda/kernels/simple.cuh"

CudaBackendV1::CudaBackendV1(const SimSnapshot &s)
    : params(s.params),
      matrix_size(s.params.pixel_size.x + 2, s.params.pixel_size.y + 2),
      redblack_matrix_size(matrix_size.x, matrix_size.y / 2),

      imax(s.params.pixel_size.x),
      jmax(s.params.pixel_size.y),
      del_x(s.params.del_x()),
      del_y(s.params.del_y()),
      ibound(s.get_boundary_cell_count()),
      ifluid(imax * jmax - ibound),

      u(matrix_size),
      v(matrix_size),

      f(matrix_size),
      g(matrix_size),

      p(matrix_size),
      p_sum_squares(matrix_size),

      p_beta(matrix_size),

      rhs(matrix_size),
      flag(matrix_size),
      fluidmask(matrix_size),
      surroundmask(matrix_size)
{
    u.memcpy_in(s.velocity_x);
    v.memcpy_in(s.velocity_y);
    p.joined.memcpy_in(s.pressure);
    flag.memcpy_in(s.get_legacy_cell_flags());

    rhs.zero_out();

    f.zero_out();
    g.zero_out();

    cudaStreamCreate(&stream);

    // Split pressure to red/black in preparation for poisson, which only operates on split matrices
    OriginalOptimized::splitToRedBlack(p.joined.as_cpu(),
                                       p.red.as_cpu(), p.black.as_cpu(),
                                       imax, jmax);

    // TODO - remove poisson_error_threshold from args
    OriginalOptimized::calculatePBeta(p_beta.joined.as_cpu(), flag.as_cpu(),
                                      imax, jmax, del_x, del_y,
                                      params.poisson_error_threshold, params.poisson_omega);
    OriginalOptimized::splitToRedBlack(p_beta.joined.as_cpu(),
                                       p_beta.red.as_cpu(), p_beta.black.as_cpu(),
                                       imax, jmax);

    // Calculate the fluidmask and surroundedmask items
    OriginalOptimized::calculateFluidmask((int**)fluidmask.as_cpu(), (const char**)flag.as_cpu(), imax, jmax);
    OriginalOptimized::splitFluidmaskToSurroundedMask((const int **)(fluidmask.as_cpu()),
                                                      (int**)surroundmask.red.as_cpu(), (int**)surroundmask.black.as_cpu(),
                                                      imax, jmax);
}

CudaBackendV1::~CudaBackendV1() {
    cudaStreamDestroy(stream);
}


float CudaBackendV1::findMaxTimestep() {
    float delta_t = -1;
    OriginalOptimized::setTimestepInterval(&delta_t,
                        imax, jmax,
                        del_x, del_y,
                        u.as_cpu(), v.as_cpu(),
                        params.Re,
                        params.timestep_safety
                        );
    DASSERT(delta_t != -1);
    return delta_t;
}

void CudaBackendV1::tick(float timestep) {
    auto gpu_params = CommonParams{
            .size = ulong2{matrix_size.x, matrix_size.y},
            .redblack_size = ulong2{redblack_matrix_size.x, redblack_matrix_size.y},
            .col_pitch_4byte=u.col_pitch,
            .col_pitch_redblack=rhs.red.col_pitch,
            .deltas = float2{del_x, del_y},
            .timestep = timestep,
    };
    dim3 blocksize_2d(16, 16);
    dim3 gridsize_2d(
            (matrix_size.x + blocksize_2d.x - 1) / blocksize_2d.x,
            (matrix_size.y + blocksize_2d.y - 1) / blocksize_2d.y
            );

    dim3 blocksize_redblack(16, 16);
    dim3 gridsize_redblack(
            (redblack_matrix_size.x + blocksize_redblack.x - 1) / blocksize_redblack.x,
            (redblack_matrix_size.y + blocksize_redblack.y - 1) / blocksize_redblack.y
    );

    dim3 blocksize_vertical(32);
    dim3 gridsize_vertical((matrix_size.y + blocksize_vertical.x - 1) / blocksize_vertical.x);

    dim3 blocksize_horizontal(32);
    dim3 gridsize_horizontal((matrix_size.x + blocksize_horizontal.x - 1) / blocksize_horizontal.x);

    computeTentativeVelocity_apply<<<gridsize_2d, blocksize_2d, 0, stream>>>(
            u.as_gpu(), v.as_gpu(), fluidmask.as_gpu(),
            f.as_gpu(), g.as_gpu(),
            gpu_params, params.gamma, params.Re
            );

    computeTentativeVelocity_postproc_vertical<<<gridsize_vertical, blocksize_vertical, 0, stream>>>(u.as_gpu(), f.as_gpu(), gpu_params);
    computeTentativeVelocity_postproc_horizontal<<<gridsize_horizontal, blocksize_horizontal, 0, stream>>>(v.as_gpu(), g.as_gpu(), gpu_params);

//    OriginalOptimized::computeRhs(f.as_cpu(), g.as_cpu(), rhs2.as_cpu(), flag.as_cpu(),
//               imax, jmax, timestep, del_x, del_y);

    computeRHS_1per<<<gridsize_2d, blocksize_2d, 0, stream>>>(f.as_gpu(), g.as_gpu(), fluidmask.as_gpu(), rhs.joined.as_gpu(), gpu_params);
    //cudaStreamSynchronize(stream);


    float res = 0;
    if (ifluid > 0) {
        constexpr bool UseCPUPoisson = false;
        if (UseCPUPoisson) {
            OriginalOptimized::poissonSolver<false>(p.joined.as_cpu(), p.red.as_cpu(), p.black.as_cpu(),
                                                    p_beta.joined.as_cpu(), p_beta.red.as_cpu(), p_beta.black.as_cpu(),
                                                    rhs.joined.as_cpu(), rhs.red.as_cpu(), rhs.black.as_cpu(),
                                                    (int **) fluidmask.as_cpu(), (int **) surroundmask.black.as_cpu(),
                                                    flag.as_cpu(), imax, jmax,
                                                    del_x, del_y,
                                                    params.poisson_error_threshold, params.poisson_max_iterations, params.poisson_omega,
                                                    ifluid);
        } else {
            // Sum of squares of pressure - reduction
            // poisson_pSquareSumReduce(p.joined.as_gpu(), p_sum_squares.as_gpu())
            // p0 = p_sum_squares.as_cpu(?????)???
            // TODO - accessing memory like this is very convenient with managed memory
            //  We *might* be able to us VK_EXT_external_memory_host to import CUDA Managed Memory as Vulkan, bypassing Vulkan allocations

            //const float partial_res_sqr_thresh = params.poisson_error_threshold * p0 * params.poisson_error_threshold * p0 * (float)ifluid;

            // Split RHS
            dispatch_splitRedBlackCUDA(rhs, gridsize_2d, blocksize_2d, gpu_params);
            // [NO CUDA STREAM SYNC NECESSARY]
            // cudaStreamSynchronize(stream);

            // Red/Black SOR-iteration
            for (int iter = 0; iter < params.poisson_max_iterations; iter++) {
            //  redblack<Red>();
                dispatch_poissonRedBlackCUDA<RedBlack::Red>(blocksize_redblack, gridsize_redblack, iter, gpu_params);
            //  float approxRes = redblack<Black>(); (capture approximate residual here)
                //float approxRes; // TODO - ???
                dispatch_poissonRedBlackCUDA<RedBlack::Black>(blocksize_redblack, gridsize_redblack, iter, gpu_params);//&approxRes);
            //  [ IMPLICIT STREAM SYNC FOR RESIDUAL ]
                // [ NOT NECESSARY WHEN NOT CALCULATING RESIDUAL ]
            //  if (approxRes < partial_res_sqr_thresh)
            //      TODO - necessary to capture full res at all? if the approxRes is actually accurate, then maybe not
            //       If we have to calculate this we may have to merge pressure here
            //      break;
            //  TODO - dynamic error
            }

            // join p
            dispatch_joinRedBlackCUDA(p, gridsize_2d, blocksize_2d, gpu_params);
            // Stream sync not necessary here, because the rest is CUDA
        }
    }

//    OriginalOptimized::updateVelocity(u.as_cpu(), v.as_cpu(),
//                       f.as_cpu(), g.as_cpu(),
//                       p.as_cpu(), flag.as_cpu(),
//                       imax, jmax, timestep, del_x, del_y);
    updateVelocity_1per<<<gridsize_2d, blocksize_2d, 0, stream>>>(f.as_gpu(), g.as_gpu(), p.joined.as_gpu(), fluidmask.as_gpu(),
                                                                      u.as_gpu(), v.as_gpu(),
                                                                      gpu_params);

    boundaryConditions_preproc_vertical<<<gridsize_vertical, blocksize_vertical, 0, stream>>>( u.as_gpu(),  v.as_gpu(), gpu_params);
    boundaryConditions_preproc_horizontal<<<gridsize_horizontal, blocksize_horizontal, 0, stream>>>( u.as_gpu(),  v.as_gpu(), gpu_params);

    boundaryConditions_apply<<<gridsize_2d, blocksize_2d, 0, stream>>>( flag.as_gpu(),
                                                                           u.as_gpu(),  v.as_gpu(),
                                                                           gpu_params);

    boundaryConditions_inputflow_west_vertical<<<gridsize_vertical, blocksize_vertical, 0, stream>>>(
            u.as_gpu(),  v.as_gpu(),
            float2{params.initial_velocity_x, params.initial_velocity_y},
            gpu_params
            );

//    OriginalOptimized::applyBoundaryConditions(u2.as_cpu(), v2.as_cpu(), flag.as_cpu(), imax, jmax, params.initial_velocity_x, params.initial_velocity_y);
}

void CudaBackendV1::dispatch_splitRedBlackCUDA(CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined>& to_split,
                                               dim3 gridsize_2d, dim3 blocksize_2d,
                                               CommonParams params)
{
    split_redblack_simple<<<gridsize_2d, blocksize_2d, 0, stream>>>(
            to_split.joined.as_gpu(),
            to_split.red.as_gpu(), to_split.black.as_gpu(),
            params
    );
}
void CudaBackendV1::dispatch_joinRedBlackCUDA(CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined>& to_join,
                                              dim3 gridsize_2d, dim3 blocksize_2d,
                                              CommonParams params)
{
    join_redblack_simple<<<gridsize_2d, blocksize_2d,0, stream>>>(
            to_join.red.as_gpu(), to_join.black.as_gpu(),
            to_join.joined.as_gpu(),
            params
    );
}

template<RedBlack Kind>
void CudaBackendV1::dispatch_poissonRedBlackCUDA(dim3 gridsize_redblack, dim3 blocksize_redblack, int iter, CommonParams gpu_params) {
    // TODO - Use HALF SIZE dimensions! the poisson kernel operates on redblack ONLY

    poisson_single_tick<<<gridsize_redblack, blocksize_redblack, 0, stream>>>(
            p.get<Kind>().as_gpu(), //out_matrix<float> this_pressure_rb,
            p.get_other<Kind>().as_gpu(),//in_matrix<float> other_pressure_rb,
            rhs.get<Kind>().as_gpu(),//in_matrix<float> this_rhs_rb,
            p_beta.get<Kind>().as_gpu(), //in_matrix<float> this_beta_rb,

            (Kind == RedBlack::Black) ? 1 : 0, // 0 if red, 1 if black

            params.poisson_omega,

            iter,

            gpu_params
                );
}

LegacySimDump CudaBackendV1::dumpStateAsLegacy() {
    cudaStreamSynchronize(stream);
    auto dump = LegacySimDump(params.to_legacy());
    dump.u = u.extract_data();
    dump.v = v.extract_data();
    dump.p = p.joined.extract_data();
    dump.flag = flag.extract_data();
    return dump;
}
SimSnapshot CudaBackendV1::get_snapshot() {
    return SimSnapshot::from_legacy(params, dumpStateAsLegacy());
}
