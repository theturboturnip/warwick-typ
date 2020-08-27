//
// Created by samuel on 12/08/2020.
//

#include "CudaBackendV1.cuh"
#include <simulation/backends/cuda/kernels/redblack.cuh>

#include "simulation/backends/original/simulation.h"
#include "simulation/backends/cuda/kernels/simple.cuh"

inline float host_min(float x, float y) {
    return (x<y) ? x : y;
}

inline float host_max(float x, float y) {
    return (x>y) ? x : y;
}

template<bool UnifiedMemory>
CudaBackendV1<UnifiedMemory>::CudaBackendV1(SimulationAllocs allocs, const FluidParams& params, const SimSnapshot& s)
    : params(params),
      simSize(s.simSize),
      matrix_size(simSize.pixel_size.x + 2, simSize.pixel_size.y + 2),
      redblack_matrix_size(matrix_size.x, matrix_size.y / 2),

      imax(simSize.pixel_size.x),
      jmax(simSize.pixel_size.y),
      x_length(simSize.physical_size.x),
      y_length(simSize.physical_size.y),
      del_x(simSize.del_x()),
      del_y(simSize.del_y()),
      ibound(s.get_boundary_cell_count()),
      ifluid(imax * jmax - ibound),

      u(allocs.alloc, allocs.u),
      v(allocs.alloc, allocs.v),

      f(allocs.alloc, matrix_size),
      g(allocs.alloc, matrix_size),

      p(allocs.alloc, allocs.p),
      p_buffered(allocs.alloc, matrix_size),
      p_sum_squares(allocs.alloc, matrix_size),

      p_beta(allocs.alloc, matrix_size),

      rhs(allocs.alloc, matrix_size),
      flag(allocs.alloc, matrix_size),
      fluidmask(allocs.alloc, allocs.fluidmask),
      surroundmask(allocs.alloc, matrix_size),

      reducer_fullsize(allocs.alloc, u.raw_length)
{
    flag.memcpy_in(s.get_legacy_cell_flags());

    rhs.zero_out();

    f.zero_out();
    g.zero_out();

    cudaStreamCreate(&stream);

    // TODO - Remove this
    //static_assert(UnifiedMemory, "CUDA backend currently requires UnifiedMemory for initial work to function");

    // Split pressure to red/black in preparation for poisson, which only operates on split matrices
    OriginalOptimized::splitToRedBlack(p.joined.as_cpu(),
                                       p.red.as_cpu(), p.black.as_cpu(),
                                       imax, jmax);
    p_buffered.red.memcpy_in(p.red);
    p_buffered.black.memcpy_in(p.black);


    // TODO - remove poisson_error_threshold from args
    OriginalOptimized::calculatePBeta(p_beta.joined.as_cpu(), flag.as_cpu(),
                                      imax, jmax, del_x, del_y,
                                      params.poisson_error_threshold, params.poisson_omega);
    OriginalOptimized::splitToRedBlack(p_beta.joined.as_cpu(),
                                       p_beta.red.as_cpu(), p_beta.black.as_cpu(),
                                       imax, jmax);

    // Calculate the fluidmask and surroundedmask items
//    OriginalOptimized::calculateFluidmask((int**)fluidmask.as_cpu(), (const char**)flag.as_cpu(), imax, jmax);
    OriginalOptimized::splitFluidmaskToSurroundedMask((const int **)(fluidmask.as_cpu()),
                                                      (int**)surroundmask.red.as_cpu(), (int**)surroundmask.black.as_cpu(),
                                                      imax, jmax);

    int dstDevice = -1;
    cudaGetDevice(&dstDevice);// TODO
    cudaDeviceProp thisDevice;
    cudaGetDeviceProperties(&thisDevice, dstDevice);
    printf("device num: %d device name: %s\n", dstDevice, thisDevice.name);
    u.dispatch_gpu_prefetch(dstDevice, stream);
    v.dispatch_gpu_prefetch(dstDevice, stream);
    p.dispatch_gpu_prefetch(dstDevice, stream);
    p_buffered.dispatch_gpu_prefetch(dstDevice, stream);
    rhs.dispatch_gpu_prefetch(dstDevice, stream);
    f.dispatch_gpu_prefetch(dstDevice, stream);
    g.dispatch_gpu_prefetch(dstDevice, stream);
    p_beta.dispatch_gpu_prefetch(dstDevice, stream);

    cudaStreamSynchronize(stream);
}

template<bool UnifiedMemory>
CudaBackendV1<UnifiedMemory>::~CudaBackendV1() {
    cudaStreamDestroy(stream);
}


template<bool UnifiedMemory>
float CudaBackendV1<UnifiedMemory>::findMaxTimestep() {
    float delta_t = -1;
    auto fabsf_lambda = [] __device__ (float x) { return fabsf(x); };
    auto max_lambda = [] __device__ (float x, float y) { return max(x, y); };
    // TODO - having multiple reducers here would be more efficient - could dispatch both, and then wait for one then the other?
    float u_max = reducer_fullsize.map_reduce(u, fabsf_lambda, max_lambda, stream);
    u_max = host_max(u_max, 1.0e-10);
    float v_max = reducer_fullsize.map_reduce(v, fabsf_lambda, max_lambda, stream);
    v_max = host_max(v_max, 1.0e-10);

    float delt_u = del_x/u_max;
    float delt_v = del_y/v_max;
    // This used to be deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;
    // the original version has 2.0 at the end, but this only ends up doing the rest of the equation, promoting it to double, dividing it, and demoting back to int.
    // this is equivalent to dividing by 2.0f without any double-promotions.
    float deltRe = 1.0f/(1.0f/(del_x*del_x)+1.0f/(del_y*del_y))*params.Re/2.0f;

    if (delt_u<delt_v) {
        delta_t = host_min(delt_u, deltRe);
    } else {
        delta_t = host_min(delt_v, deltRe);
    }
    delta_t = params.timestep_safety * (delta_t); // multiply by safety factor

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

template<bool UnifiedMemory>
void CudaBackendV1<UnifiedMemory>::tick(float timestep) {
    auto gpu_params = CommonParams{
            .size = uint2{matrix_size.x, matrix_size.y},
            .redblack_size = uint2{redblack_matrix_size.x, redblack_matrix_size.y},
            .col_pitch_4byte=u.col_pitch,
            .col_pitch_redblack=rhs.red.col_pitch,
            .deltas = float2{del_x, del_y},
            .timestep = timestep,
    };
    dim3 blocksize_2d(1, 64);
    dim3 gridsize_2d(
            (matrix_size.x + blocksize_2d.x - 1) / blocksize_2d.x,
            (matrix_size.y + blocksize_2d.y - 1) / blocksize_2d.y
            );

    dim3 blocksize_redblack(1, 64);
    dim3 gridsize_redblack(
            (redblack_matrix_size.x + blocksize_redblack.x - 1) / blocksize_redblack.x,
            (redblack_matrix_size.y + blocksize_redblack.y - 1) / blocksize_redblack.y
    );
    //printf("blksize_redblack: %d %d, gridsize: %d %d\n", blocksize_redblack.x, blocksize_redblack.y, gridsize_redblack.x, gridsize_redblack.y);

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


    if (ifluid > 0) {
        constexpr bool UseCPUPoisson = false;
        if constexpr (UnifiedMemory && UseCPUPoisson) {
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
                dispatch_poissonRedBlackCUDA<RedBlack::Red>(gridsize_redblack, blocksize_redblack, iter, gpu_params);
            //  float approxRes = redblack<Black>(); (capture approximate residual here)
                //float approxRes; // TODO - ???
                dispatch_poissonRedBlackCUDA<RedBlack::Black>(gridsize_redblack, blocksize_redblack, iter, gpu_params);//&approxRes);
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

template<bool UnifiedMemory>
void CudaBackendV1<UnifiedMemory>::dispatch_splitRedBlackCUDA(CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined>& to_split,
                                               dim3 gridsize_2d, dim3 blocksize_2d,
                                               CommonParams params)
{
    split_redblack_simple<<<gridsize_2d, blocksize_2d, 0, stream>>>(
            to_split.joined.as_gpu(),
            to_split.red.as_gpu(), to_split.black.as_gpu(),
            params
    );
}
template<bool UnifiedMemory>
void CudaBackendV1<UnifiedMemory>::dispatch_joinRedBlackCUDA(CudaUnifiedRedBlackArray<float, UnifiedMemory, RedBlackStorage::WithJoined>& to_join,
                                              dim3 gridsize_2d, dim3 blocksize_2d,
                                              CommonParams params)
{
    join_redblack_simple<<<gridsize_2d, blocksize_2d,0, stream>>>(
            to_join.red.as_gpu(), to_join.black.as_gpu(),
            to_join.joined.as_gpu(),
            params
    );
}

template<bool UnifiedMemory>
template<RedBlack Kind>
void CudaBackendV1<UnifiedMemory>::dispatch_poissonRedBlackCUDA(dim3 gridsize_redblack, dim3 blocksize_redblack, int iter, CommonParams gpu_params) {
    // TODO - Use HALF SIZE dimensions! the poisson kernel operates on redblack ONLY

    // For a p_red computation: do p_red/p_buffered_black into p_buffered_red, while copying p_buffered_black into p_black.
    // Modern Nvidia GPUs can do parallel memcpy and compute, so this shouldn't take longer

    // Note - all p.get<Kind> functions need to have the template specifier: "p.template get<Kind>()".
    // This is because:
    //  1. CudaBackendV1 is templated on UnifiedMemory
    //  2. Which causes the types of p, p_buffered, rhs etc. to be *dependent* on the value of UnifiedMemory.
    //  3. Because of template specializations, the C++ compiler can't know ahead of time if p.get will be a template function or a data member.
    //  4. This means the C++ compiler can't tell what p.get<Kind>() means - is it (p.get) < Kind > ()? or is it a template function invocation?
    //  5. Using the '.template' syntax tells the C++ compiler to explicitly expect a template function here. It's ugly as hell, but it works.
    // The way to fix this is to stop using shitty languages like C++. However, given the state of CUDA on better languages like Rust is kinda poor,
    // this isn't feasible right now.

    constexpr bool DoubleBuffer = false;

    if (DoubleBuffer) {
        poisson_single_tick<<<gridsize_redblack, blocksize_redblack, 0, stream>>>(
                p.template get<Kind>().as_gpu(),
                p_buffered.template get_other<Kind>().as_gpu(),
                rhs.template get<Kind>().as_gpu(),
                p_beta.template get<Kind>().as_gpu(),

                p_buffered.template get<Kind>().as_gpu(),

                (Kind == RedBlack::Black) ? 1 : 0,// 0 if red, 1 if black

                params.poisson_omega,

                iter,

                gpu_params);

        // TODO - this needs to be done in a separate stream to overlap
        p.template get_other<Kind>().dispatch_memcpy_in(p_buffered.template get_other<Kind>(), stream);
    } else {
        poisson_single_tick<<<gridsize_redblack, blocksize_redblack, 0, stream>>>(
                p.template get<Kind>().as_gpu(),
                p.template get_other<Kind>().as_gpu(),
                rhs.template get<Kind>().as_gpu(),
                p_beta.template get<Kind>().as_gpu(),

                p.template get<Kind>().as_gpu(),

                (Kind == RedBlack::Black) ? 1 : 0,// 0 if red, 1 if black

                params.poisson_omega,

                iter,

                gpu_params);
    }

//    cudaError_t error = (cudaPeekAtLastError());
//    if (error != cudaSuccess) {
//        FATAL_ERROR("CUDA ERROR %s\n", cudaGetErrorString(error));
//    }
}

template<>
LegacySimDump CudaBackendV1<true>::dumpStateAsLegacy() {
    cudaStreamSynchronize(stream);
    auto dump = LegacySimDump(simSize.to_legacy());
    dump.u = u.extract_data();
    dump.v = v.extract_data();
    dump.p = p.joined.extract_data();
    dump.flag = flag.extract_data();
    return dump;
}
//template<>
//LegacySimDump CudaBackendV1<false>::dumpStateAsLegacy() {
//    static_assert(false, "Cannot dump state, not using Unified Memory");
//}
template<bool UnifiedMemory>
SimSnapshot CudaBackendV1<UnifiedMemory>::get_snapshot() {
    return SimSnapshot::from_legacy(dumpStateAsLegacy());
}


template class CudaBackendV1<true>;
template class CudaBackendV1<false>;