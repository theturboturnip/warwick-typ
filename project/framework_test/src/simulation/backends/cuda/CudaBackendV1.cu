//
// Created by samuel on 12/08/2020.
//

#include "CudaBackendV1.cuh"

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
      p_red(redblack_matrix_size),
      p_black(redblack_matrix_size),

      p_beta(matrix_size),
      p_beta_red(redblack_matrix_size),
      p_beta_black(redblack_matrix_size),

      rhs(matrix_size),
      rhs_red(redblack_matrix_size),
      rhs_black(redblack_matrix_size),

      flag(matrix_size),
      fluidmask(matrix_size),
      surroundmask_red(redblack_matrix_size),
      surroundmask_black(redblack_matrix_size)
{
    u.memcpy_in(s.velocity_x);
    v.memcpy_in(s.velocity_y);
    p.memcpy_in(s.pressure);
    flag.memcpy_in(s.get_legacy_cell_flags());

    rhs.zero_out();
    rhs_red.zero_out();
    rhs_black.zero_out();

    f.zero_out();
    g.zero_out();

    cudaStreamCreate(&stream);

    // TODO - remove poisson_error_threshold from args
    OriginalOptimized::calculatePBeta(p_beta.as_cpu(), flag.as_cpu(), imax, jmax, del_x, del_y, params.poisson_error_threshold, params.poisson_omega);
    OriginalOptimized::splitToRedBlack(p.as_cpu(), p_red.as_cpu(), p_black.as_cpu(), imax, jmax);
    OriginalOptimized::splitToRedBlack(p_beta.as_cpu(), p_beta_red.as_cpu(), p_beta_black.as_cpu(), imax, jmax);
    OriginalOptimized::calculateFluidmask((int**)fluidmask.as_cpu(), (const char**)flag.as_cpu(), imax, jmax);
    OriginalOptimized::splitFluidmaskToSurroundedMask((const int **)(fluidmask.as_cpu()), (int**)surroundmask_red.as_cpu(), (int**)surroundmask_black.as_cpu(), imax, jmax);
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
    //const int ifluid = (imax * jmax) - ibound;
    auto gpu_params = CommonParams{
            .size = ulong2{matrix_size.x, matrix_size.y},
            .deltas = float2{del_x, del_y},
            .timestep = timestep,
    };
    dim3 threads_per_block(32, 32);
    dim3 num_blocks(
            (matrix_size.x + threads_per_block.x - 1) / threads_per_block.x,
            (matrix_size.y + threads_per_block.y - 1) / threads_per_block.y
            );

    OriginalOptimized::computeTentativeVelocity(u.as_cpu(), v.as_cpu(), f.as_cpu(), g.as_cpu(), flag.as_cpu(),
                             imax, jmax, timestep, del_x, del_y, params.gamma, params.Re);
//    OriginalOptimized::computeRhs(f.as_cpu(), g.as_cpu(), rhs.as_cpu(), flag.as_cpu(),
//               imax, jmax, timestep, del_x, del_y);
//    cudaMemPrefetchAsync(f.raw_data, rhs.width*rhs.height*sizeof(float), 0, stream);
//    cudaMemPrefetchAsync(g.raw_data, rhs.width*rhs.height*sizeof(float), 0, stream);
//    cudaMemPrefetchAsync(fluidmask.raw_data, rhs.width*rhs.height*sizeof(int), 0, stream);
//    cudaStreamSynchronize(stream);
    computeRHS_1per<<<num_blocks, threads_per_block, 0, stream>>>(f.as_gpu().constify(), g.as_gpu().constify(), fluidmask.as_gpu().constify(),
                          rhs.as_gpu(), gpu_params);
    cudaStreamSynchronize(stream);
//    cudaMemPrefetchAsync(rhs.raw_data, rhs.width*rhs.height*sizeof(float), cudaCpuDeviceId, stream);

    float res = 0;
    if (ifluid > 0) {
        OriginalOptimized::poissonSolver<false>(p.as_cpu(), p_red.as_cpu(), p_black.as_cpu(),
                                                p_beta.as_cpu(), p_beta_red.as_cpu(), p_beta_black.as_cpu(),
                                                rhs.as_cpu(), rhs_red.as_cpu(), rhs_black.as_cpu(),
                                                (int**)fluidmask.as_cpu(), (int**)surroundmask_black.as_cpu(),
                                                flag.as_cpu(), imax, jmax,
                                                del_x, del_y,
                                                params.poisson_error_threshold, params.poisson_max_iterations, params.poisson_omega,
                                                ifluid);
    }

    OriginalOptimized::updateVelocity(u.as_cpu(), v.as_cpu(),
                       f.as_cpu(), g.as_cpu(),
                       p.as_cpu(), flag.as_cpu(),
                       imax, jmax, timestep, del_x, del_y);
    OriginalOptimized::applyBoundaryConditions(u.as_cpu(), v.as_cpu(), flag.as_cpu(), imax, jmax, params.initial_velocity_x, params.initial_velocity_y);
}
LegacySimDump CudaBackendV1::dumpStateAsLegacy() {
    auto dump = LegacySimDump(params.to_legacy());
    dump.u = u.extract_data();
    dump.v = v.extract_data();
    dump.p = p.extract_data();
    dump.flag = flag.extract_data();
    return dump;
}
SimSnapshot CudaBackendV1::get_snapshot() {
    return SimSnapshot::from_legacy(params, dumpStateAsLegacy());
}
