//
// Created by samuel on 13/08/2020.
//

#include "simple.cuh"

#include "simulation/backends/original/constants.h"

#include <cstdio>

// TODO - the current paradigm for kernels is to return early, but this prevents __syncthreads() from working.
//  https://stackoverflow.com/a/6667067
//  In a non-cooperative context this is fine, and all of these kernels are for embarassingly parallel tasks with no intra-task dependencies,
//  but if any of these were to be made cooperative then this would need to be considered.

// This is used to simulate a fma() C call - nvcc should optimize this down to a single FMA instr
__device__ inline float fma_cuda(float a, float b, float c) {
    //return a * b + c;

    // _rn = round to nearest, which is the default.
    return __fmaf_rn(a, b, c);
}

__global__ void computeTentativeVelocity_apply(
        in_matrix<float> u, in_matrix<float> v, in_matrix<uint> is_fluid,
        out_matrix<float> f, out_matrix<float> g,
        const CommonParams params,
        const float timestep,
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
    const uint idx_south_east = params.flatten_4byte(i+1, j-1);
    const uint idx_north_west = params.flatten_4byte(i-1, j+1);

    // TODO - Pass these in as arguments
    // laplu/laplv use double precision literals in the original code, so are calculated at double precision, but then
    // are rounded down to single precision.
    // at that precision, the difference between (1/delx/delx) and (1/(delx*delx)) is very small, especially at double precision
    // adding the div by Re makes it faster, but puts accuracy down to e=0.0001
    const float delx2 = 1.0f/(params.deltas.x * params.deltas.x);
    const float dely2 = 1.0f/(params.deltas.y * params.deltas.y);

    // The use of `double fabs(double);` in du2dx, duvdy etc. force the division by 4*dely to be performed at double precision
    // However, the result is rounded down to single precision directly afterwards.
    // This means we can multiply by the reciporical instead without any loss in accuracy on the given input data.
    const float _4delx = 1.0f/(4.0f*params.deltas.x);
    const float _4dely = 1.0f/(4.0f*params.deltas.y);

    // TODO - could be worth splitting into two kernels for idx_east, idx_south?
    // In large majority of cases both will be true - only false when directly on a boundary
    // This means the % of warps that diverge here is small, so this should be OK.
//    for (i=1; i<=imax-1; i++) { <- we should reject i >= imax == params.size.x - 2
//        for (j=1; j<=jmax; j++) {
    if ((i < params.size.x - 2) && is_fluid[idx_east]) {
        float du2dx = ((u[idx]+u[idx_east])*(u[idx]+u[idx_east])+
                       gamma*fabsf(u[idx]+u[idx_east])*(u[idx]-u[idx_east])-
                       (u[idx_west]+u[idx])*(u[idx_west]+u[idx])-
                       gamma*fabsf(u[idx_west]+u[idx])*(u[idx_west]-u[idx]))
                      *_4delx;
//        float du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
//                       gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
//                       (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
//                       gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
//                      *_4delx;
        float duvdy = ((v[idx]+v[idx_east])*(u[idx]+u[idx_north])+
                       gamma*fabsf(v[idx]+v[idx_east])*(u[idx]-u[idx_north])-
                       (v[idx_south]+v[idx_south_east])*(u[idx_south]+u[idx])-
                       gamma*fabsf(v[idx_south]+v[idx_south_east])*(u[idx_south]-u[idx]))
                      *_4dely;

        float laplu = fma_cuda((fma_cuda(-2.0f, u[idx], u[idx_east])+u[idx_west]), delx2,
                          (fma_cuda(-2.0f, u[idx], u[idx_north])+u[idx_south])*dely2);


        // This is not implicitly casted, so the division by Re cannot be converted to a multiplication.
        f[idx] = u[idx]+timestep*(laplu/Re-du2dx-duvdy);
//
//        if (i == 100 && j == 2) {
//            printf("100/2 GPU f\n");
//            printf("%.9g %.9g %.9g\n", u[idx_west], u[idx], u[idx_east]);
//            printf("%.9g %.9g %.9g\n", du2dx, duvdy, laplu);
//            printf("%.9g %.9g\n", _4delx, gamma);
//            printf("%.9g = %.9g %.9g\n", f[idx], u[idx], params.timestep);
//        }
    } else {
        f[idx] = u[idx];
    }

//    for (i=1; i<=imax; i++) {
//        for (j=1; j<=jmax-1; j++) { <- we should reject j >= jmax == params.size.y - 2
    if ((j < params.size.y - 2) && is_fluid[idx_north]) {
        float duvdx = ((u[idx]+u[idx_north])*(v[idx]+v[idx_east])+
                       gamma*fabsf(u[idx]+u[idx_north])*(v[idx]-v[idx_east])-
                       (u[idx_west]+u[idx_north_west])*(v[idx_west]+v[idx])-
                       gamma*fabsf(u[idx_west]+u[idx_north_west])*(v[idx_west]-v[idx]))
                      *_4delx;
        float dv2dy = ((v[idx]+v[idx_north])*(v[idx]+v[idx_north])+
                       gamma*fabsf(v[idx]+v[idx_north])*(v[idx]-v[idx_north])-
                       (v[idx_south]+v[idx])*(v[idx_south]+v[idx])-
                       gamma*fabsf(v[idx_south]+v[idx])*(v[idx_south]-v[idx]))
                      *_4dely;

        float laplv = fma_cuda((fma_cuda(-2.0f, v[idx], v[idx_east])+v[idx_west]),delx2,
                          (fma_cuda(-2.0f, v[idx], v[idx_north])+v[idx_south])*dely2);

//        if (i == 100 && j == 2) {
//            printf("100/2 GPU g\n");
////            printf("%a %a %a\n", u[idx_west], u[idx], u[idx_east]);
//            printf("%a %a %a\n", duvdx, dv2dy, laplv);
//            printf("%a %a\n", _4delx, gamma);
//        }

        g[idx] = v[idx]+timestep*(laplv/Re-duvdx-dv2dy);
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

    const uint idx_0 = params.flatten_4byte(i, 0);
    g[idx_0] = v[idx_0];

    // TODO - why is this at jmax? there's another element just after
    const uint idx_jmax = params.flatten_4byte(i, params.size.y - 2);
    g[idx_jmax] = v[idx_jmax];
}

__global__ void computeRHS_1per(in_matrix<float> f, in_matrix<float> __restrict__ g, in_matrix<uint> is_fluid, out_matrix<float> rhs,
                                const CommonParams params, const float timestep) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (!params.in_real_range(i, j)) return;

    const uint idx = params.flatten_4byte(i, j);

    if (!is_fluid[idx]) return;

    const float f_this = f[idx];
    const float f_west = f[params.flatten_4byte(i - 1, j)];
    const float g_this = g[idx];
    const float g_south = g[params.flatten_4byte(i, j - 1)];

    const float new_rhs = ((f_this-f_west)/params.deltas.x + (g_this-g_south)/params.deltas.y) / timestep;

//    if (i == 100 && j == 2) {
//        printf("GPU RHS %dx%d\n", i, j);
//
//        printf("f: %.9g\t%.9g\n", f_this, f_west);
//        printf("g: %.9g\t%.9g\n", g_this, g_south);
//        printf("new_rhs: %.9g\n", new_rhs);
//        printf("dx: %.9g\tdy: %.9g\tdt: %.9g\n", params.deltas.x, params.deltas.y, params.timestep);
//    }

    rhs[idx] = new_rhs;
}

__global__ void poisson_single_tick(in_matrix<float> this_pressure_rb,
                                    in_matrix<float> other_pressure_rb,
                                    in_matrix<float> this_rhs_rb,
                                    in_matrix<float> this_beta_rb,
                                    out_matrix<float> this_pressure_rb_out,

                                    int is_black, // 0 if red, 1 if black

                                    float poisson_omega,

                                    int iter,

                                    const CommonParams params
) {

//    // If the column is even, then the first red is p[i][0], which shouldn't be calculated, so start at index 1 in the red array
//    const int j_start = (i % 2 == rb) ? 1 : 0;
//    // In even columns, the "north" value is at p_red[i][j-1], but on odd columns, the "north" value is p_red[i][j].
//    const int north_offset = -j_start;

    // These are redblack offsets - not for normal matrices
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    //if (!params.in_real_range(i, j)) return;
    // i params are the same in redblack as they are in normal ones
    if (i == 0 || i >= params.size.x - 1) return;
    // j params have a start point based on i:
    //  (0, 0) in absolute coords is red. (0,0) in absolute coords should not be processed, becuase it's on an edge.
    //  this applies to all items on a horizontal edge - (0,0), (1,0), (2,0), (3,0), (4,0) in absolute coords shouldn't be processed.
    //  if we're red, then on even columns, the top item (2n, 0) is on the edge, and we should start processing at 1.
    //  if we're black, then on odd columns, the top item (2n+1, 0) is on the edge, and we should start processing at 1.
    const int j_start = (i % 2 == is_black) ? 1 : 0;
    // redblack_size.y = (jmax+2)/2 = (jmax/2) + 1
    // (jmax/2) items must be processed
    // => if j_start == 0, don't process elements >= (jmax/2) = redblack_size.y - 1 + 0
    //      if j_start == 1, don't process elements >= (jmax/2) + 1 = redblack_size.y - 1 + 1
    if (j < j_start || j >= (params.redblack_size.y - 1 + j_start)) return;

    // if we're red, the position of the corresponding "south" square in the *other* array is dependent on the column just like j_start.
    // In even columns, the "south" value is at p_red[i][j-1], but on odd columns, the "south" value is p_red[i][j].
    const int south_offset_in_other = -j_start;

    // The index of our current value in *this* matrix, applicable to p, p_beta, rhs
    const uint curr_idx = params.flatten_redblack(i, j);
    // Offset the indices for the north/south values in the *other* matrix
    const uint north_idx_other = params.flatten_redblack(i, j+south_offset_in_other+1);
    const uint south_idx_other = params.flatten_redblack(i, j+south_offset_in_other);
    // In adjacent columns in the *other* matrix, j-values are equivalent
    const uint east_idx_other = params.flatten_redblack(i+1, j);
    const uint west_idx_other = params.flatten_redblack(i-1, j);

    // These reads are not contiguous with each other, but they should be contiguous with the other accesses in this warp
//    const float north = other_pressure_rb[north_idx_other];
//    const float south = other_pressure_rb[south_idx_other];
//    const float east = other_pressure_rb[east_idx_other];
//    const float west = other_pressure_rb[west_idx_other];
    const float north = other_pressure_rb[curr_idx+south_offset_in_other+1];
    const float south = other_pressure_rb[curr_idx+south_offset_in_other];
    const float east = other_pressure_rb[curr_idx + params.col_pitch_redblack];
    const float west = other_pressure_rb[curr_idx - params.col_pitch_redblack];

    const float centre = this_pressure_rb[curr_idx];
    const float beta = this_beta_rb[curr_idx];
    const float rhs = this_rhs_rb[curr_idx];

//    __m128 north = _mm_loadu_ps(&updown_col[j+north_offset]);
//    __m128 south = _mm_loadu_ps(&updown_col[j+north_offset+1]);
//    __m128 east = _mm_loadu_ps(&right_col[j]);
//    __m128 west = _mm_loadu_ps(&left_col[j]);
//
//    __m128 centre = _mm_loadu_ps(&mid_col[j]);
//
//    __m128 beta_mod = _mm_loadu_ps(&mid_beta[j]);
//
//    __m128 rhs = _mm_loadu_ps(&mid_rhs[j]);
//

    // TODO - pass as params
    const float rdx2 = 1.0f/(params.deltas.x*params.deltas.x);
    const float rdy2 = 1.0f/(params.deltas.y*params.deltas.y);
    const float inv_omega = 1.0f - poisson_omega;

    const float horiz = (east + west) * rdx2;
    const float vertical = (north + south) * rdy2;

    const float sum = beta * (horiz + vertical - rhs);

    // On CPU this is an FMSUB, fma of negative should translate to a proper GPU fmsub
    const float final = fma_cuda(inv_omega, centre, (-sum));//(inv_omega * centre) - sum;

//    if ((i == 100) && j == (0 + j_start) && iter == 0 && is_black == 0) {
//        printf("GPU REPORT %d %dx%d\n", is_black, i, j);
//
//        printf("n: %a\ts: %a\te: %a\tw: %a\n",
//               (north),
//               (south),
//               (east),
//               (west)
//        );
//
//        printf("c: %a\tbeta: %a\trhs: %a\n",
//               (centre),
//               (beta),
//               (rhs)
//        );
//
//        printf("rdx2: %a\trdy2: %a\tinv_omega: %a\n",
//               (rdx2),
//               (rdy2),
//               (inv_omega)
//        );
//
//        printf("horiz: %a\tvertical: %a\tsum: %a\n",
//               (horiz),
//               (vertical),
//               (sum)
//        );
//
//        printf("final: %a\n", (final));
//    }
    
    this_pressure_rb_out[curr_idx] = final;
//    __m128 horiz = _mm_add_ps(east, west);
//    __m128 vertical = _mm_add_ps(north, south);
//    //
//
//    horiz = _mm_mul_ps(horiz, rdx2_v);
//    vertical = _mm_mul_ps(vertical, rdy2_v);
//    __m128 sum = _mm_mul_ps(beta_mod, _mm_sub_ps(_mm_add_ps(horiz, vertical), rhs));
//
//    __m128 final = _mm_fmsub_ps(inv_omega_v, centre, sum);
//
//    _mm_storeu_ps(&mid_col[j], final);
}

__global__ void updateVelocity_1per(in_matrix<float> f, in_matrix<float> g, in_matrix<float> p, in_matrix<uint> is_fluid,
                                    out_matrix<float> u, out_matrix<float> v,
                                    const CommonParams params,
                                    const float timestep)
{
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (!params.in_real_range(i, j)) return;

    const uint idx = params.flatten_4byte(i, j);

    if (is_fluid[idx]) {
        const uint idx_east = params.flatten_4byte(i+1, j);
        const uint idx_north = params.flatten_4byte(i, j+1);

        if (is_fluid[idx_east]) {
            u[idx] = f[idx]-(p[idx_east]-p[idx])*timestep/params.deltas.x;
        }
        if (is_fluid[idx_north]) {
            v[idx] = g[idx]-(p[idx_north]-p[idx])*timestep/params.deltas.y;
        }
    }
}

__global__ void boundaryConditions_preproc_vertical(out_matrix<float> u, out_matrix<float> v, const CommonParams params){
    const uint j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (j >= params.size.y) return;

    const uint idx = params.flatten_4byte(0, j);
    const uint idx_east = params.flatten_4byte(1, j);

    // Fluid freely flows in from the west
    u[idx] = u[idx_east];
    v[idx] = v[idx_east];

    // Fluid freely flows out to the east
    // TODO - this is weird - why not use imax+1, imax for u?
//    u[imax][j] = u[imax-1][j];
//    v[imax+1][j] = v[imax][j];
    const uint idx_imax_west = params.flatten_4byte(params.size.x-3, j);
    const uint idx_imax = params.flatten_4byte(params.size.x-2, j);
    const uint idx_imax_east = params.flatten_4byte(params.size.x-1, j);
    u[idx_imax] = u[idx_imax_west];
    v[idx_imax_east] = v[idx_imax];
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

    if (j >= params.size.y - 1) return;

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