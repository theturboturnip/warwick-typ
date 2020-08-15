//
// Created by samuel on 15/08/2020.
//

#include "redblack.cuh"

__global__ void split_redblack_simple(in_matrix<float> joined,
                                      out_matrix<float> red, out_matrix<float> black,
                                      const CommonParams params) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= params.size.x) return;
    if (j >= params.size.y) return;

//    for (i = 0; i < imax+2; i++) {
//        for (j = 0; j < jmax+2; j++) {
//            if ((i+j) % 2 == 0)
//                red[i][j >> 1] = joined[i][j];
//            else
//                black[i][j >> 1] = joined[i][j];
//        }
//    }

    const uint idx_joined = params.flatten_4byte(i, j);
    const uint idx_redblack = params.flatten_redblack(i, j >> 1u);
    // TODO - equivalent to (i&1) ^ (j&2) == 0 - does the compiler figure this out
    if ((i+j) % 2 == 0)
        red[idx_redblack] = joined[idx_joined];
    else
        black[idx_redblack] = joined[idx_joined];
}

__global__ void join_redblack_simple(in_matrix<float> red, in_matrix<float> black,
                                     out_matrix<float> joined,
                                     const CommonParams params) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= params.size.x) return;
    if (j >= params.size.y) return;

//    for (i = 0; i < imax+2; i++) {
//        for (j = 0; j < jmax+2; j++) {
//            if ((i+j) % 2 == 0)
//                joined[i][j] = red[i][j >> 1];
//            else
//                joined[i][j] = black[i][j >> 1];
//        }
//    }

    const uint idx_joined = params.flatten_4byte(i, j);
    const uint idx_redblack = params.flatten_redblack(i, j >> 1u);
    // TODO - equivalent to (i&1) ^ (j&2) == 0 - does the compiler figure this out
    if ((i+j) % 2 == 0)
        joined[idx_joined] = red[idx_redblack];
    else
        joined[idx_joined] = black[idx_redblack];
}