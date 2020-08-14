//
// Created by samuel on 13/08/2020.
//

#include "simple.cuh"

#include "simulation/backends/original/constants.h"

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

__global__ void boundaryConditions_preproc_vertical(out_matrix<float> u, out_matrix<float> v, const CommonParams params){
    const uint j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (j >= params.size.y) return;

    const uint idx = params.flatten_4byte(0, j);
    const uint idx_west = params.flatten_4byte(1, j);

    // Fluid freely flows in from the west
    u[idx] = u[idx_west];
    v[idx] = v[idx_west];

    // Fluid freely flows out to the east
    // TODO - this is weird - why not use imax+1, imax for u?
//    u[imax][j] = u[imax-1][j];
//    v[imax+1][j] = v[imax][j];
    const uint idx_imax_west = params.flatten_4byte(params.size.x-1, j);
    const uint idx_imax = params.flatten_4byte(params.size.x-2, j);
    const uint idx_imax_east = params.flatten_4byte(params.size.x-3, j);
    u[idx_imax] = u[idx_imax_east];
    v[idx_imax_west] = v[idx_imax];
}
__global__ void boundaryConditions_preproc_horizontal(out_matrix<float> u, out_matrix<float> v, const CommonParams params){
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= params.size.x) return;


    /* The vertical velocity approaches 0 at the north and south
     * boundaries, but fluid flows freely in the horizontal direction */
//    v[i][0] = 0.0;
//    v[i][jmax] = 0.0;
//
//    u[i][0] = u[i][1];
//    u[i][jmax+1] = u[i][jmax];

    const uint idx_i_0 = params.flatten_4byte(i, 0);
    const uint idx_i_1 = params.flatten_4byte(i, 1);
    const uint idx_i_jmax = params.flatten_4byte(i, params.size.y-2);
    const uint idx_i_jmax_1 = params.flatten_4byte(i, params.size.y-1);

    u[idx_i_0] = u[idx_i_1];
    u[idx_i_jmax_1] = u[idx_i_jmax];

    v[idx_i_0] = 0;
    v[idx_i_jmax] = 0;
}

__global__ void boundaryConditions_apply(in_matrix<char> flag,
                                         out_matrix<float> u, out_matrix<float> v, const CommonParams params) {
    /* Apply no-slip boundary conditions to cells that are adjacent to
 * internal obstacle cells. This forces the u and v velocity to
 * tend towards zero in these cells.
 */
//#pragma omp parallel for schedule(static) private(j) shared(u, v, flag, imax, jmax) default(none)
//    for (i=1; i<=imax; i++) {
//        for (j=1; j<=jmax; j++) {
//            if (flag[i][j] & B_NSEW) {
//                ...

    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (!params.in_real_range(i, j)) return;

    const uint idx = params.flatten_4byte(i, j);

    if (!(flag[idx] & B_NSEW)) return;

    //printf("%03d %03d\n", i, j);

    const uint idx_east = params.flatten_4byte(i+1, j);
    const uint idx_west = params.flatten_4byte(i-1, j);
    const uint idx_north = params.flatten_4byte(i, j+1);
    const uint idx_south = params.flatten_4byte(i, j-1);
    const uint idx_south_west = params.flatten_4byte(i-1, j-1);
    const uint idx_south_east = params.flatten_4byte(i+1, j-1);
    const uint idx_north_west = params.flatten_4byte(i-1, j+1);

    switch (flag[idx]) {
        case B_N:
            v[idx]   = 0.0;
            u[idx]   = -u[idx_north];
            u[idx_west] = -u[idx_north_west];
            break;
        case B_E:
            u[idx]   = 0.0;
            v[idx]   = -v[idx_east];
            v[idx_south] = -v[idx_south_east];
            break;
        case B_S:
            v[idx_south] = 0.0;
            u[idx]   = -u[idx_south];
            u[idx_west] = -u[idx_south_west];
            break;
        case B_W:
            u[idx_west] = 0.0;
            v[idx]   = -v[idx_west];
            v[idx_south] = -v[idx_south_west];
            break;
        case B_NE:
            v[idx]   = 0.0;
            u[idx]   = 0.0;
            v[idx_south] = -v[idx_south_east];
            u[idx_west] = -u[idx_north_west];
            break;
        case B_SE:
            v[idx_south] = 0.0;
            u[idx]   = 0.0;
            v[idx]   = -v[idx_east];
            u[idx_west] = -u[idx_south_west];
            break;
        case B_SW:
            v[idx_south] = 0.0;
            u[idx_west] = 0.0;
            v[idx]   = -v[idx_west];
            u[idx]   = -u[idx_south];
            break;
        case B_NW:
            v[idx]   = 0.0;
            u[idx_west] = 0.0;
            v[idx_south] = -v[idx_south_west];
            u[idx]   = -u[idx_north];
            break;
    }
}

__global__ void boundaryConditions_inputflow_west_vertical(out_matrix<float> u, out_matrix<float> v, float2 west_velocity, const CommonParams params) {
    /* Finally, fix the horizontal velocity at the  western edge to have
 * a continual flow of fluid into the simulation.
 */
//    v[0][0] = 2*vi-v[1][0];
//    for (j=1;j<=jmax;j++) {
//        u[0][j] = ui;
//        v[0][j] = 2*vi-v[1][j];
//    }

    const uint j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (j >= params.size.y) return;

    const uint idx_0_j = params.flatten_4byte(0, j);
    const uint idx_1_j = params.flatten_4byte(1, j);

    v[idx_0_j] = 2 * west_velocity.y - v[idx_1_j];

    // TODO - this is an annoying conditional, do we *need* the velocity x in the corners to be preserved?
    if (1 <= j && j <= params.size.y - 2) {
        u[idx_0_j] = west_velocity.x;
    }
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