//
// Created by samuel on 13/08/2020.
//

#include "simple.cuh"

#include "simulation/backends/original/constants.h"

#include <cstdio>

// This is used to simulate a fma() C call - nvcc should optimize this down to a single FMA instr
__device__ inline double fma_cuda(double a, double b, double c) {
    //return a * b + c;
    return __fma_rn(a, b, c);
}

__global__ void computeTentativeVelocity_apply(
        in_matrix<float> u, in_matrix<float> v, in_matrix<uint> is_fluid,
        out_matrix<float> f, out_matrix<float> g,
        const CommonParams params,
        const float gamma, const float Re
) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (!params.in_real_range(i, j)) return;

    const uint idx = params.flatten_4byte(i, j);

    if (!is_fluid[idx]) return;

    const uint idx_east = params.flatten_4byte(i+1, j);
    const uint idx_west = params.flatten_4byte(i-1, j);
    const uint idx_north = params.flatten_4byte(i, j+1);
    const uint idx_south = params.flatten_4byte(i, j-1);
    const uint idx_south_west = params.flatten_4byte(i-1, j-1);
    const uint idx_south_east = params.flatten_4byte(i+1, j-1);
    const uint idx_north_west = params.flatten_4byte(i-1, j+1);

    // TODO - Pass these in as arguments
    // laplu/laplv use double precision literals in the original code, so are calculated at double precision, but then
    // are rounded down to single precision.
    // at that precision, the difference between (1/delx/delx) and (1/(delx*delx)) is very small, especially at double precision
    // adding the div by Re makes it faster, but puts accuracy down to e=0.0001
    const double delx2 = 1.0/((double)params.deltas.x * (double)params.deltas.x);
    const double dely2 = 1.0/((double)params.deltas.y * (double)params.deltas.y);

    // The use of `double fabs(double);` in du2dx, duvdy etc. force the division by 4*dely to be performed at double precision
    // However, the result is rounded down to single precision directly afterwards.
    // This means we can multiply by the reciporical instead without any loss in accuracy on the given input data.
    const double _4delx = 1.0/(4.0*params.deltas.x);
    const double _4dely = 1.0/(4.0*params.deltas.y);

    // TODO - could be worth splitting into two kernels for idx_east, idx_south?
    // In large majority of cases both will be true - only false when directly on a boundary
//    for (i=1; i<=imax-1; i++) { <- we should reject i >= imax == params.size.x - 2
//        for (j=1; j<=jmax; j++) {
    if ((i < params.size.x - 2) && is_fluid[idx_east]) {
        // TODO Fix floating point issues - use single only?
        float du2dx = ((u[idx]+u[idx_east])*(u[idx]+u[idx_east])+
                       gamma*fabs(u[idx]+u[idx_east])*(u[idx]-u[idx_east])-
                       (u[idx_west]+u[idx])*(u[idx_west]+u[idx])-
                       gamma*fabs(u[idx_west]+u[idx])*(u[idx_west]-u[idx]))
                      *_4delx;
        float duvdy = ((v[idx]+v[idx_east])*(u[idx]+u[idx_north])+
                       gamma*fabs(v[idx]+v[idx_east])*(u[idx]-u[idx_north])-
                       (v[idx_south]+v[idx_south_east])*(u[idx_south]+u[idx])-
                       gamma*fabs(v[idx_south]+v[idx_south_east])*(u[idx_south]-u[idx]))
                      *_4dely;

        float laplu = fma_cuda((fma_cuda(-2.0, u[idx], u[idx_east])+u[idx_west]), delx2,
                          (fma_cuda(-2.0, u[idx], u[idx_north])+u[idx_south])*dely2);

        // This is not implicitly casted, so the division by Re cannot be converted to a multiplication.
        f[idx] = u[idx]+params.timestep*(laplu/Re-du2dx-duvdy);
    } else {
        f[idx] = u[idx];
    }

//    for (i=1; i<=imax; i++) {
//        for (j=1; j<=jmax-1; j++) { <- we should reject j >= jmax == params.size.y - 2
    if ((j < params.size.y - 2) && is_fluid[idx_north]) {
        // TODO Fix floating point issues as above
        float duvdx = ((u[idx]+u[idx_north])*(v[idx]+v[idx_east])+
                       gamma*fabs(u[idx]+u[idx_north])*(v[idx]-v[idx_east])-
                       (u[idx_west]+u[idx_north_west])*(v[idx_west]+v[idx])-
                       gamma*fabs(u[idx_west]+u[idx_north_west])*(v[idx_west]-v[idx]))
                      *_4delx;
        float dv2dy = ((v[idx]+v[idx_north])*(v[idx]+v[idx_north])+
                       gamma*fabs(v[idx]+v[idx_north])*(v[idx]-v[idx_north])-
                       (v[idx_south]+v[idx])*(v[idx_south]+v[idx])-
                       gamma*fabs(v[idx_south]+v[idx])*(v[idx_south]-v[idx]))
                      *_4dely;

        float laplv = fma_cuda((fma_cuda(-2.0, v[idx], v[idx_east])+v[idx_west]),delx2,
                          (fma_cuda(-2.0, v[idx], v[idx_north])+v[idx_south])*dely2);

        g[idx] = v[idx]+params.timestep*(laplv/Re-duvdx-dv2dy);
    } else {
        g[idx] = v[idx];
    }
}

__global__ void computeTentativeVelocity_postproc_vertical(in_matrix<float> u, out_matrix<float> f, const CommonParams params) {
    const uint j = (blockIdx.x * blockDim.x) + threadIdx.x;

    // for (j=1; j<=jmax; j++) {
    if (j == 0 || j >= params.size.y - 1) return;

//        f[0][j]    = u[0][j];
//        f[imax][j] = u[imax][j];
//    }

    const uint idx_0 = params.flatten_4byte(0, j);
    f[idx_0] = u[idx_0];

    // TODO - why is this at imax? there's another element just after
    const uint idx_imax = params.flatten_4byte(params.size.x - 2, j);
    f[idx_imax] = u[idx_imax];
}
__global__ void computeTentativeVelocity_postproc_horizontal(in_matrix<float> v, out_matrix<float> g, const CommonParams params) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // for (i=1; i<=imax; i++) {
    if (i == 0 || i >= params.size.x - 1) return;

//        g[i][0]    = v[i][0];
//        g[i][jmax] = v[i][jmax];
//    }

    const uint idx_0 = params.flatten_4byte(1, 0);
    g[idx_0] = v[idx_0];

    // TODO - why is this at jmax? there's another element just after
    const uint idx_jmax = params.flatten_4byte(i, params.size.y - 2);
    g[idx_jmax] = v[idx_jmax];
}

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