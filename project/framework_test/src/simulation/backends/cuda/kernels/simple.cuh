//
// Created by samuel on 13/08/2020.
//

#pragma once

#include "common.cuh"

__global__ void computeTentativeVelocity_apply(
        in_matrix<float> u, in_matrix<float> v, in_matrix<uint> is_fluid,
        out_matrix<float> f, out_matrix<float> g,
        const CommonParams params,
        const float gamma, const float Re
        );

__global__ void computeTentativeVelocity_postproc_vertical(in_matrix<float> u, out_matrix<float> f, const CommonParams params);
__global__ void computeTentativeVelocity_postproc_horizontal(in_matrix<float> v, out_matrix<float> g, const CommonParams params);


__global__ void computeRHS_1per(in_matrix<float> f, in_matrix<float> g, in_matrix<uint> is_fluid, out_matrix<float> rhs,
                                const CommonParams params);

// TODO - this should operate linearly, as 2D traversal is never performed
// poisson_pSquareSumReduce(in_matrix<float> p, out_matrix<float> p_squares)
__global__ void poisson_single_tick(out_matrix<float> this_pressure_rb,
                                    in_matrix<float> other_pressure_rb,
                                    in_matrix<float> this_rhs_rb,
                                    in_matrix<float> this_beta_rb,

                                    int is_black, // 0 if red, 1 if black
                                    float poisson_omega,
                                    const CommonParams params
                                    );


__global__ void updateVelocity_1per(in_matrix<float> f, in_matrix<float> g, in_matrix<float> p, in_matrix<uint> is_fluid,
                                    out_matrix<float> u, out_matrix<float> v,
                                    const CommonParams params);

__global__ void boundaryConditions_preproc_vertical(out_matrix<float> u, out_matrix<float> v, const CommonParams params);
__global__ void boundaryConditions_preproc_horizontal(out_matrix<float> u, out_matrix<float> v, const CommonParams params);

__global__ void boundaryConditions_apply(in_matrix<char> flags, out_matrix<float> u, out_matrix<float> v, const CommonParams params);

__global__ void boundaryConditions_inputflow_west_vertical(out_matrix<float> u, out_matrix<float> v, float2 west_velocity, const CommonParams params);

__global__ void set(float* output, float val, CommonParams params);