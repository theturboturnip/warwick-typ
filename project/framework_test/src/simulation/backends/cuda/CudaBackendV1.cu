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

    // TODO - remove poisson_error_threshold from args
    OriginalOptimized::calculatePBeta(p_beta.joined.as_cpu(), flag.as_cpu(),
                                      imax, jmax, del_x, del_y,
                                      params.poisson_error_threshold, params.poisson_omega);
    OriginalOptimized::splitToRedBlack(p.joined.as_cpu(),
                                       p.red.as_cpu(), p.black.as_cpu(),
                                       imax, jmax);
    OriginalOptimized::splitToRedBlack(p_beta.joined.as_cpu(),
                                       p_beta.red.as_cpu(), p_beta.black.as_cpu(),
                                       imax, jmax);
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
            .col_pitch_4byte=u.col_pitch,
            .col_pitch_redblack=rhs.red.col_pitch,
            .deltas = float2{del_x, del_y},
            .timestep = timestep,
    };
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
            (matrix_size.x + threads_per_block.x - 1) / threads_per_block.x,
            (matrix_size.y + threads_per_block.y - 1) / threads_per_block.y
            );

    dim3 vertical_blocksize(32);
    dim3 vertical_num_blocks((matrix_size.y + vertical_blocksize.x - 1) / vertical_blocksize.x);

    dim3 horizontal_blocksize(32);
    dim3 horizontal_num_blocks((matrix_size.x + horizontal_blocksize.x - 1) / horizontal_blocksize.x);

    computeTentativeVelocity_apply<<<num_blocks, threads_per_block, 0, stream>>>(
            u.as_gpu(), v.as_gpu(), fluidmask.as_gpu(),
            f.as_gpu(), g.as_gpu(),
            gpu_params, params.gamma, params.Re
            );

    computeTentativeVelocity_postproc_vertical<<<vertical_num_blocks, vertical_blocksize, 0, stream>>>(u.as_gpu(), f.as_gpu(), gpu_params);
    computeTentativeVelocity_postproc_horizontal<<<horizontal_num_blocks, horizontal_blocksize, 0, stream>>>(v.as_gpu(), g.as_gpu(), gpu_params);

//    OriginalOptimized::computeRhs(f.as_cpu(), g.as_cpu(), rhs2.as_cpu(), flag.as_cpu(),
//               imax, jmax, timestep, del_x, del_y);

    computeRHS_1per<<<num_blocks, threads_per_block, 0, stream>>>(f.as_gpu(), g.as_gpu(), fluidmask.as_gpu(), rhs.joined.as_gpu(), gpu_params);
    cudaStreamSynchronize(stream);


    float res = 0;
    if (ifluid > 0) {
        OriginalOptimized::poissonSolver<false>(p.joined.as_cpu(), p.red.as_cpu(), p.black.as_cpu(),
                                                p_beta.joined.as_cpu(), p_beta.red.as_cpu(), p_beta.black.as_cpu(),
                                                rhs.joined.as_cpu(), rhs.red.as_cpu(), rhs.black.as_cpu(),
                                                (int**)fluidmask.as_cpu(), (int**)surroundmask.black.as_cpu(),
                                                flag.as_cpu(), imax, jmax,
                                                del_x, del_y,
                                                params.poisson_error_threshold, params.poisson_max_iterations, params.poisson_omega,
                                                ifluid);
    }

//    OriginalOptimized::updateVelocity(u.as_cpu(), v.as_cpu(),
//                       f.as_cpu(), g.as_cpu(),
//                       p.as_cpu(), flag.as_cpu(),
//                       imax, jmax, timestep, del_x, del_y);
    updateVelocity_1per<<<num_blocks, threads_per_block, 0, stream>>>(f.as_gpu(), g.as_gpu(), p.joined.as_gpu(), fluidmask.as_gpu(),
                                                                      u.as_gpu(), v.as_gpu(),
                                                                      gpu_params);

    boundaryConditions_preproc_vertical<<<vertical_num_blocks, vertical_blocksize, 0, stream>>>( u.as_gpu(),  v.as_gpu(), gpu_params);
    boundaryConditions_preproc_horizontal<<<horizontal_num_blocks, horizontal_blocksize, 0, stream>>>( u.as_gpu(),  v.as_gpu(), gpu_params);

    boundaryConditions_apply<<<num_blocks, threads_per_block, 0, stream>>>( flag.as_gpu(),
                                                                           u.as_gpu(),  v.as_gpu(),
                                                                           gpu_params);

    boundaryConditions_inputflow_west_vertical<<<vertical_num_blocks, vertical_blocksize, 0, stream>>>(
            u.as_gpu(),  v.as_gpu(),
            float2{params.initial_velocity_x, params.initial_velocity_y},
            gpu_params
            );

//    OriginalOptimized::applyBoundaryConditions(u2.as_cpu(), v2.as_cpu(), flag.as_cpu(), imax, jmax, params.initial_velocity_x, params.initial_velocity_y);
}

void CudaBackendV1::dispatch_splitRedBlackCUDA(CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined>& to_split,
                                               dim3 blocksize_2d, dim3 gridsize_2d,
                                               CommonParams params)
{
    split_redblack_simple<<<gridsize_2d, blocksize_2d, 0, stream>>>(
            to_split.joined.as_gpu(),
            to_split.red.as_gpu(), to_split.black.as_gpu(),
            params
    );
}
void CudaBackendV1::dispatch_joinRedBlackCUDA(CudaUnifiedRedBlackArray<float, RedBlackStorage::WithJoined>& to_join,
                                              dim3 blocksize_2d, dim3 gridsize_2d,
                                              CommonParams params)
{
    join_redblack_simple<<<gridsize_2d, blocksize_2d,0, stream>>>(
            to_join.red.as_gpu(), to_join.black.as_gpu(),
            to_join.joined.as_gpu(),
            params
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
