//
// Created by samuel on 13/08/2020.
//

#pragma once

#include "common.cuh"


__global__ void computeRHS_1per(Array2D<const float> f, Array2D<const float> g, Array2D<const uint> flag,
                                Array2D<float> rhs,
                                CommonParams params);