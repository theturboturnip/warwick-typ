//
// Created by samuel on 13/08/2020.
//

#pragma once

#include "common.cuh"

template<typename T>
using in_matrix = const T* const __restrict__;

template<typename T>
using out_matrix = T* const __restrict__;


__global__ void computeRHS_1per(in_matrix<float> f, in_matrix<float> __restrict__ g, in_matrix<uint> is_fluid, out_matrix<float> rhs,
                                const CommonParams params);

__global__ void updateVelocity_1per(in_matrix<float> f, in_matrix<float> g, in_matrix<float> p, in_matrix<uint> is_fluid,
                                    out_matrix<float> u, out_matrix<float> v,
                                    const CommonParams params);

__global__ void computeRHS_1per(//Array2D<const float> f, Array2D<const float> g, Array2D<const uint> flag,
                                //Array2D<float> rhs,
         const float* __restrict__ f, const float* __restrict__ g, const uint* __restrict__ is_fluid, float* rhs,
                                CommonParams params);

__global__ void set(float* output, float val, CommonParams params);