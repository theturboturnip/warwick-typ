//
// Created by samuel on 15/08/2020.
//

#pragma once

#include "common.cuh"

__global__ void split_redblack_simple(in_matrix<float> joined,
                                      out_matrix<float> red, out_matrix<float> black,
                                      const CommonParams params);

__global__ void join_redblack_simple(in_matrix<float> red, in_matrix<float> black,
                                     out_matrix<float> joined,
                                     const CommonParams params);