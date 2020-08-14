//
// Created by samuel on 13/08/2020.
//

#include "simple.cuh"

#include <cstdio>

__global__ void computeRHS_1per(in_matrix<float> f, in_matrix<float> __restrict__ g, in_matrix<uint> is_fluid, out_matrix<float> rhs,
                                const CommonParams params) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (!params.in_real_range(i, j)) return;

    const uint idx = params.flatten_4byte(i, j);

    if (!is_fluid[idx]) return;

    const float f_this = f[idx];
    const float f_west = f[params.flatten_4byte(i - 1, j)];
    const float g_this = g[idx];
    const float g_south = g[params.flatten_4byte(i, j - 1)];

    const float new_rhs = ((f_this-f_west)/params.deltas.x + (g_this-g_south)/params.deltas.y) / params.timestep;

    rhs[idx] = new_rhs;
}

__global__ void updateVelocity_1per(in_matrix<float> f, in_matrix<float> g, in_matrix<float> p, in_matrix<uint> is_fluid,
                                    out_matrix<float> u, out_matrix<float> v,
                                    const CommonParams params)
{

    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (!params.in_real_range(i, j)) return;

    const uint idx = params.flatten_4byte(i, j);

    if (is_fluid[idx]) {
        const uint idx_east = params.flatten_4byte(i+1, j);
        const uint idx_north = params.flatten_4byte(i, j+1);

        if (is_fluid[idx_east]) {
            u[idx] = f[idx]-(p[idx_east]-p[idx])*params.timestep/params.deltas.x;
        }
        if (is_fluid[idx_north]) {
            v[idx] = g[idx]-(p[idx_north]-p[idx])*params.timestep/params.deltas.y;
        }
    }

    // Loop was fused and parallelized
//#pragma omp parallel for schedule(static) private(j) default(none)
//    for (i=1; i<=imax; i++) {
//        for (j=1; j<=jmax; j++) {
//            // only if both adjacent cells are fluid cells
//            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
//                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
//            }
//
//            // only if both adjacent cells are fluid cells
//            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
//                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
//            }
//        }
//    }
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