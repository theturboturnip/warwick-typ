//
// Created by samuel on 13/08/2020.
//

#include "simple.cuh"

#include <cstdio>

__global__ void computeRHS_1per(//const Array2D<const float> f, const Array2D<const float> g, const Array2D<const uint> is_fluid,
                                //Array2D<float> rhs,
        const float* __restrict__ f, const float* __restrict__ g, const uint* __restrict__ is_fluid, float* rhs,
                                const CommonParams p) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    // for i = [1, imax] inclusive
    // p.size.x = imax+2
    // so if i == imax + 1 == p.size.x - 1, ignore
    // p.size.y = 122 if jmax = 120 => j = [1, 120] inclusive
    // we shouldn't touch the perimiter: i == 0, i == p.size.x - 1 = 661
    // j == 0, j == p.size.y - 1 = 121
    // i,j can also go beyond those limits if p.size is not a multiple of 16 => use >= instead of ==
    if ((i == 0) || (i >= p.size.x - 1)) return;
    if ((j == 0) || (j >= p.size.y - 1)) return;

    if (!is_fluid[i * p.size.y + j]) return;

    const float f_this = f[i * p.size.y + j];
    const float f_west = f[(i-1) * p.size.y + j];
    const float g_this = g[i * p.size.y + j];
    const float g_south = g[i * p.size.y + (j-1)];

    const float new_rhs = ((f_this-f_west)/p.deltas.x + (g_this-g_south)/p.deltas.y) / p.timestep;

    rhs[i * p.size.y + j] = new_rhs;
}

__global__ void set(float* output, float val, CommonParams p) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    const uint idx = i * p.size.y + j;

    if ((i == 0) || (i >= p.size.x - 1)) return;
    if ((j == 0) || (j >= p.size.y - 1)) return;

//    printf("%03d, %03d (%03lu %03lu) -> %p[%5d] -> %f\n", i, j, p.size.x, p.size.y, output, idx, val);
//    printf("(was %f) \n", output[idx]);

    output[idx] = val;
}