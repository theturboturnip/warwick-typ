#include "simulation.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "constants.h"

#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

namespace OriginalOptimized {

// Computation of tentative velocity field (f, g)
template<typename Float>
void computeTentativeVelocity(const float ** const u, const float ** const v, float ** const f, float ** const g,
                              char ** const flag, const int imax, const int jmax, const float del_t, const float delx, const float dely,
                              const float gamma, const float Re);

template<>
void computeTentativeVelocity<double>(const float ** const u, const float ** const v, float ** const f, float ** const g,
                                      char ** const flag, const int imax, const int jmax, const float del_t, const float delx, const float dely,
                                      const float gamma, const float Re)
{
    int  i, j;

    // laplu/laplv use double precision literals in the original code, so are calculated at double precision, but then
    // are rounded down to single precision.
    // at that precision, the difference between (1/delx/delx) and (1/(delx*delx)) is very small, especially at double precision
    // adding the div by Re makes it faster, but puts accuracy down to e=0.0001
    const double delx2 = 1.0/((double)delx * (double)delx);
    const double dely2 = 1.0/((double)dely * (double)dely);

    // The use of `double fabs(double);` in du2dx, duvdy etc. force the division by 4*dely to be performed at double precision
    // However, the result is rounded down to single precision directly afterwards.
    // This means we can multiply by the reciporical instead without any loss in accuracy on the given input data.
    const double _4delx = 1.0/(4.0*delx);
    const double _4dely = 1.0/(4.0*dely);

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                float du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                         gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                         (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                         gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                        *_4delx;
                float duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                         gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                         (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                         gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                        *_4dely;

                float laplu = fma((fma(-2.0, u[i][j], u[i+1][j])+u[i-1][j]), delx2,
                            (fma(-2.0, u[i][j], u[i][j+1])+u[i][j-1])*dely2);

                // This is not implicitly casted, so the division by Re cannot be converted to a multiplication.
                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                float duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                         gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                         (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                         gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                        *_4delx;
                float dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                         gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                         (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                         gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                        *_4dely;

                float laplv = fma((fma(-2.0, v[i][j], v[i+1][j])+v[i-1][j]),delx2,
                            (fma(-2.0, v[i][j], v[i][j+1])+v[i][j-1])*dely2);

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    // f & g at external boundaries
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (i=1; i<=imax; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}

template<>
void computeTentativeVelocity<float>(const float ** const u, const float ** const v, float ** const f, float ** const g,
                                     char ** const flag, const int imax, const int jmax, const float del_t, const float delx, const float dely,
                                     const float gamma, const float Re)
{
    int  i, j;

    const float delx2 = 1.0f/(delx * delx);
    const float dely2 = 1.0f/(dely * dely);

    // The use of `double fabs(double);` in du2dx, duvdy etc. force the division by 4*dely to be performed at double precision
    // However, the result is rounded down to single precision directly afterwards.
    // This means we can multiply by the reciporical instead without any loss in accuracy on the given input data.
    const float _4delx = 1.0f/(4.0f*delx);
    const float _4dely = 1.0f/(4.0f*dely);

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                float du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                               gamma*fabsf(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                               (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                               gamma*fabsf(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                              *_4delx;
                float duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                               gamma*fabsf(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                               (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                               gamma*fabsf(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                              *_4dely;

                float laplu = fmaf((fmaf(-2.0f, u[i][j], u[i+1][j])+u[i-1][j]), delx2,
                                  (fmaf(-2.0f, u[i][j], u[i][j+1])+u[i][j-1])*dely2);

//                if (i == 100 && j == 100) {
//                    printf("100/100 CPU\n");
//                    //printf("%.9g %.9g %.9g\n", u[i-1][j], u[i][j], u[i+1][j]);
//                    printf("%a %a %a\n", du2dx, duvdy, laplu);
//                    //printf("%.9g %.9g\n", _4delx, gamma);
//                }


                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
//                f[i][j] = fmaf(del_t, (laplu/Re-du2dx-duvdy), u[i][j]);
//
//                if (i == 100 && j == 2) {
//                    printf("100/2 CPU f\n");
//                    printf("%.9g %.9g %.9g\n", u[i-1][j], u[i][j], u[i+1][j]);
//                    printf("%.9g %.9g %.9g\n", du2dx, duvdy, laplu);
//                    printf("%.9g %.9g\n", _4delx, gamma);
//                    printf("%.9g = %.9g %.9g\n", f[i][j], u[i][j], del_t);
//                }
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                float duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                               gamma*fabsf(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                               (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                               gamma*fabsf(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                              *_4delx;
                float dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                               gamma*fabsf(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                               (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                               gamma*fabsf(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                              *_4dely;

                float laplv = fmaf((fmaf(-2.0f, v[i][j], v[i+1][j])+v[i-1][j]),delx2,
                                  (fmaf(-2.0f, v[i][j], v[i][j+1])+v[i][j-1])*dely2);

//                if (i == 100 && j == 2) {
//                    printf("100/2 CPU g\n");
////                    printf("%a %a %a\n", u[idx_west], u[idx], u[idx_east]);
//                    printf("%a %a %a\n", duvdx, dv2dy, laplv);
//                    printf("%a %a\n", _4delx, gamma);
//                }

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
//                g[i][j] = fmaf(del_t, (laplv/Re-duvdx-dv2dy), v[i][j]);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    // f & g at external boundaries
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (i=1; i<=imax; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}

// Calculate the right hand side of the pressure equation
void computeRhs(float ** const f, float ** const g, float ** const rhs, char ** const flag, const int imax,
                const int jmax, const float del_t, const float delx, const float dely)
{
    int i, j;

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i=1;i<=imax;i++) {
        for (j=1;j<=jmax;j++) {
            if (flag[i][j] & C_F) {
                // only for fluid and non-surface cells
                rhs[i][j] = (
                                    (f[i][j]-f[i-1][j])/delx +
                                    (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
//                if (i == 100 && j == 2) {
//                    printf("CPU RHS %dx%d\n", i, j);
//
//                    printf("f: %.9g\t%.9g\n", f[i][j], f[i-1][j]);
//                    printf("g: %.9g\t%.9g\n", g[i][j], g[i][j-1]);
//                    printf("new_rhs: %.9g\n", rhs[i][j]);
//                    printf("dx: %.9g\tdy: %.9g\tdt: %.9g\n", delx, dely, del_t);
//                }
            }
        }
    }
}


// Red/Black SOR to solve the poisson equation
template<bool ErrorCheck>
int poissonSolver(float ** const p, float ** const p_red, float ** const p_black,
                  float ** const p_beta, float ** const p_beta_red, float ** const p_beta_black,
                  float ** const rhs, float ** const rhs_red, float ** const rhs_black,
                  int ** const fluidmask, int ** const surroundmask_black,
                  char ** const flag, const int imax, const int jmax,
                  const float delx, const float dely, const float eps, const int itermax, const float omega,
                  const int ifull)
{
    int i, j, iter;

    int rb; // Red-black value.

    const float rdx2 = 1.0/(delx*delx);
    const __m128 rdx2_v = _mm_set1_ps(rdx2);
    const float rdy2 = 1.0/(dely*dely);
    const __m128 rdy2_v = _mm_set1_ps(rdy2);
    const float inv_omega = 1.0 - omega;
    const __m128 inv_omega_v = _mm_set1_ps(inv_omega);

    float p0 = 0.0;
    if (ErrorCheck) {
        // Calculate sum of squares
        for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    float p_val = ((i + j) % 2 == 0) ? p_red[i][j >> 1] : p_black[i][j >> 1];
                    p0 += p_val * p_val;
                }
            }
        }

        p0 = sqrt(p0 / ifull);
        if (p0 < 0.0001) { p0 = 1.0; }
    }
    // Replace usages of *res with a stack variable, this can be converted to a register by the compiler
    // This is restored to *res at the end of the program
    float res_stack = 0.0;

    // We compute a part of the res during the black iteration
    // Once that part of the res is larger than this value, we know that we'll need to perform the next iteration.
    const double partial_res_sqr_thresh = (double)eps * (double)p0 * (double)eps * (double)p0 * (double)ifull;

    // The RHS function does not operate on split matrices, so split the matrix back up
    splitToRedBlack(rhs, rhs_red, rhs_black, imax, jmax);

    // Red/Black SOR-iteration

    for (iter = 0; iter < itermax; iter++) {
        //res_stack = 0.0f;
        //fprintf(stderr, "res_stack reset: %g\n", res_stack);
        for (rb = 0; rb <= 1; rb++) {
            float threadlocal_res = 0.0f;

            float ** const this_color = rb ? p_black : p_red;
            // This could be const float ** const, but it complains about casting
            float ** const other_color = rb ? p_red : p_black;

            // This breaks res_stack - presumably the reduction keeps some internal variable which isn't reset
#pragma omp parallel for schedule(static) private(j) shared(rb, iter) firstprivate(threadlocal_res) default(none)// reduction(+:res_stack)
            for (i = 1; i <= imax; i++) {
                const float *const left_col = other_color[i-1];
                const float *const right_col = other_color[i+1];
                const float *const updown_col = other_color[i];
                float *const mid_col = this_color[i];

                const float *const mid_rhs = rb ? rhs_black[i] : rhs_red[i];
                const float *const mid_beta = rb ? p_beta_black[i] : p_beta_red[i];

                // If the column is even, then the first red is p[i][0], which shouldn't be calculated, so start at index 1 in the red array
                const int j_start = (i % 2 == rb) ? 1 : 0;
                // In even columns, the "north" value is at p_red[i][j-1], but on odd columns, the "north" value is p_red[i][j].
                const int north_offset = -j_start;

                // This is assuming jmax%2 == 0
                const int total_elements_processed = jmax/2;

#define VECTOR_LENGTH 4
                const int total_vectors_processed = total_elements_processed / VECTOR_LENGTH;
                const int last_vector_end = j_start + total_vectors_processed * VECTOR_LENGTH;

                // If we're doing the black side, and we aren't sure if we're within the error range, try adding to the residual.
                if (ErrorCheck && rb == 1 && threadlocal_res < partial_res_sqr_thresh) {
                    for(j = j_start; j < last_vector_end; j += VECTOR_LENGTH) {
                        __m128 north = _mm_loadu_ps(&updown_col[j+north_offset]);
                        __m128 south = _mm_loadu_ps(&updown_col[j+north_offset+1]);
                        __m128 east = _mm_loadu_ps(&right_col[j]);
                        __m128 west = _mm_loadu_ps(&left_col[j]);

                        __m128 centre = _mm_loadu_ps(&mid_col[j]);

                        __m128 beta_mod = _mm_loadu_ps(&mid_beta[j]);

                        __m128 rhs = _mm_loadu_ps(&mid_rhs[j]);

                        __m128 horiz = _mm_add_ps(east, west);
                        __m128 vertical = _mm_add_ps(north, south);
                        //

                        horiz = _mm_mul_ps(horiz, rdx2_v);
                        vertical = _mm_mul_ps(vertical, rdy2_v);
                        __m128 sum = _mm_mul_ps(beta_mod, _mm_sub_ps(_mm_add_ps(horiz, vertical), rhs));

                        __m128 final = _mm_fmsub_ps(inv_omega_v, centre, sum);

                        _mm_storeu_ps(&mid_col[j], final);

                        // Extra code for checking residual
                        __m128i surroundmask_v = _mm_loadu_si128((__m128i*)&surroundmask_black[i][j]);
                        __m128 adds = _mm_fmadd_ps(_mm_sub_ps(east, west), rdx2_v, _mm_fmsub_ps(_mm_sub_ps(north, south), rdy2_v, rhs));
                        adds = _mm_and_ps(adds, _mm_castsi128_ps(surroundmask_v));
                        adds = _mm_dp_ps(adds, adds, 0xFF);
                        float final_add = _mm_cvtss_f32(adds);
                        threadlocal_res += final_add;
                        //res_stack += final_add;
                    }
                    /*for(;j < j_end; j++){
                      float north = updown_col[j+north_offset];
                      float south = updown_col[j+north_offset+1];
                      float east = right_col[j];
                      float west = left_col[j];
                      float centre = mid_col[j];
                      float beta_mod = mid_beta[j];
                      float m_rhs = -mid_rhs[j];

                      float horiz = east + west;
                      float vert = north + south;

                      // rdx2, rdy2 are in the range of 500-1000, but rhs is >0.1.
                      // The beta_half of the equation is calculated at single precision
                      // which means the order of operations is delicate due to rounding.
                      // In the original, horiz*rdx2 + vert * rdy2 are summed first, then rhs is subtracted.
                      // Even without FMAs or any other tricks, performing those in any other order results in an "incorrect" simulation.
                      // Adding parentheses anywhere other than (horiz*rdx2+vert*rdy2) results in an incorrect sim.
                      // Similarly, if any FMA is used to calculate (horiz*rdx2+vert*rdy2) the simulation is "incorrect".

                      // TODO: Adding fmaf() specifically here breaks things.
                      float beta_half = -beta_mod * ((horiz * rdx2 + vert * rdy2) + m_rhs);
                      //float beta_half = -beta_mod * fmaf(horiz, rdx2, fmaf(vert, rdy2, m_rhs));
                      mid_col[j] = fmaf(inv_omega, centre, beta_half);

//mid_col[j] = fma(inv_omega, centre, (double)beta_half);

if (surroundmask_black[i][j]) {
// This is a black fluid square surrounded by other fluid squares
float add = (east - west) * rdx2 + (north - south) * rdy2 + m_rhs;
threadlocal_res += add * add;
res_stack += add * add;
}
} */ // end of j black_check
                } else {

                    // This assumes the size of the
                    for(j = j_start; j < last_vector_end; j += VECTOR_LENGTH) {
                        __m128 north = _mm_loadu_ps(&updown_col[j+north_offset]);
                        __m128 south = _mm_loadu_ps(&updown_col[j+north_offset+1]);
                        __m128 east = _mm_loadu_ps(&right_col[j]);
                        __m128 west = _mm_loadu_ps(&left_col[j]);

                        __m128 centre = _mm_loadu_ps(&mid_col[j]);

                        __m128 beta_mod = _mm_loadu_ps(&mid_beta[j]);

                        __m128 rhs = _mm_loadu_ps(&mid_rhs[j]);

                        __m128 horiz = _mm_add_ps(east, west);
                        __m128 vertical = _mm_add_ps(north, south);

                        // CUDA DISCREPANCY - In CUDA horiz*rdx2 + vertical is turned into an FMA, here it is not.
                        // The FMA was purposefully avoided in this code, but it seems like it doesn't have an impact on bit-accuracy in CUDA?
                        horiz = _mm_mul_ps(horiz, rdx2_v);
                        vertical = _mm_mul_ps(vertical, rdy2_v);
                        __m128 sum = _mm_mul_ps(beta_mod, _mm_sub_ps(_mm_add_ps(horiz, vertical), rhs));
                        // This is not contracted to FMA by default on CUDA
                        __m128 final = _mm_fmsub_ps(inv_omega_v, centre, sum);

//                        if ((i == 100) && j == (0 + j_start) && iter == 0 && rb == 0) {
//                            printf("CPU REPORT %d %dx%d\n", rb, i, j);
//
//                            printf("n: %a\ts: %a\te: %a\tw: %a\n",
//                                   _mm_cvtss_f32(north),
//                                   _mm_cvtss_f32(south),
//                                   _mm_cvtss_f32(east),
//                                   _mm_cvtss_f32(west)
//                            );
//
//                            printf("c: %a\tbeta: %a\trhs: %a\n",
//                                   _mm_cvtss_f32(centre),
//                                   _mm_cvtss_f32(beta_mod),
//                                   _mm_cvtss_f32(rhs)
//                            );
//
//                            printf("rdx2: %a\trdy2: %a\tinv_omega: %a\n",
//                                   _mm_cvtss_f32(rdx2_v),
//                                   _mm_cvtss_f32(rdy2_v),
//                                   _mm_cvtss_f32(inv_omega_v)
//                            );
//
//                            printf("horiz: %a\tvertical: %a\tsum: %a\n",
//                                   _mm_cvtss_f32(horiz),
//                                   _mm_cvtss_f32(vertical),
//                                   _mm_cvtss_f32(sum)
//                            );
//
//                            printf("final: %a\n", _mm_cvtss_f32(final));
//                        }

                        _mm_storeu_ps(&mid_col[j], final);
                    }  // end of j
                    // This is a cleanup loop.
                    // Including this is unnecessary for the given input data, as jmax is divisible by 4
                    /*for(;j < j_end; j++){
                      float north = updown_col[j+north_offset];
                      float south = updown_col[j+north_offset+1];
                      float east = right_col[j];
                      float west = left_col[j];
                      float centre = mid_col[j];
                      float beta_mod = mid_beta[j];
                      float m_rhs = -mid_rhs[j];

                      float horiz = east + west;
                      float vert = north + south;

                      // rdx2, rdy2 are in the range of 500-1000, but rhs is <0.1.
                      // The beta_half of the equation is calculated at single precision
                      // which means the order of operations is delicate due to rounding.
                      // In the original, horiz*rdx2 + vert * rdy2 are summed first, then rhs is subtracted.
                      // Even without FMAs or any other tricks, performing those in any other order results in an "incorrect" simulation.
                      // Adding parentheses anywhere other than (horiz*rdx2+vert*rdy2) results in an incorrect sim.
                      // Similarly, if any FMA is used to calculate (horiz*rdx2+vert*rdy2) the simulation is "incorrect".
                      float beta_half = -beta_mod * ((horiz * rdx2 + vert * rdy2) + m_rhs);
                      mid_col[j] = fmaf(inv_omega, centre, beta_half);
                      } */ // end of j cleanup loop
                }

            } // end of i
        } // end of rb

        //fprintf(stderr, "res_stack: %g, partial_thresh: %g\n", res_stack, partial_res_sqr_thresh);

        //if (res_stack > partial_res_sqr_thresh) continue;
        //fprintf(stderr, "Didn't skip, res_stack=%.9g < %.9g\n", res_stack, partial_res_sqr_thresh);
        if (ErrorCheck) {

            res_stack = 0.0f;
            // Compute the residual
            // TODO: Use other fluidmask bits to make it faster? Probably not needed

            // The rest of the code does not operate on split P matrices, so join them back up
            joinRedBlack(p, p_red, p_black, imax, jmax);

#pragma omp parallel for private(j) default(none) reduction(+: res_stack)
            for (i = 1; i <= imax; i++) {
                for (j = 1; j <= jmax; j++) {
                    //if ((i+j)%2 != 0) continue;
                    if (flag[i][j] & C_F) {
                        // only fluid cells
                        float add = (fluid_E_mask(p[i + 1][j] - p[i][j]) -
                                     fluid_W_mask(p[i][j] - p[i - 1][j])) *
                                            rdx2 +
                                    (fluid_N_mask(p[i][j + 1] - p[i][j]) -
                                     fluid_S_mask(p[i][j] - p[i][j - 1])) *
                                            rdy2 -
                                    rhs[i][j];
                        res_stack += add * add;
                    }
                }
            }
            res_stack = sqrt((res_stack) / ifull) / p0;
            ///fprintf(stdout, "res_stack: %g, eps: %g, ifull:%d, p0:%g\n", res_stack, eps, ifull, p0);
            if (res_stack < eps) {
                //fprintf(stdout, "\n");
                return iter;
            }
        }

//        TODO - allow dynamic error to be enabled!!!
//        if (iter % 100 == 0 && eps < 0.01f) {
//            eps *= 1.01f;
//        }
    } // end of iter


    // The rest of the code does not operate on split P matrices, so join them back up
    joinRedBlack(p, p_red, p_black, imax, jmax);
    //fprintf(stderr, "\n");

    return iter;
}
template int poissonSolver<true>(float ** const p, float ** const p_red, float ** const p_black,
                                 float ** const p_beta, float ** const p_beta_red, float ** const p_beta_black,
                                 float ** const rhs, float ** const rhs_red, float ** const rhs_black,
                                 int ** const fluidmask, int ** const surroundmask_black,
                                 char ** const flag, const int imax, const int jmax,
                                 const float delx, const float dely, const float eps, const int itermax, const float omega,
                                 const int ifull);
template int poissonSolver<false>(float ** const p, float ** const p_red, float ** const p_black,
                                  float ** const p_beta, float ** const p_beta_red, float ** const p_beta_black,
                                  float ** const rhs, float ** const rhs_red, float ** const rhs_black,
                                  int ** const fluidmask, int ** const surroundmask_black,
                                  char ** const flag, const int imax, const int jmax,
                                  const float delx, const float dely, const float eps, const int itermax, const float omega,
                                  const int ifull);

void calculatePBeta(float ** const p_beta,
                    char ** const flag,
                    const int imax, const int jmax,
                    const float delx, const float dely, const float eps, const float omega) {
    int i, j;

    const float rdx2 = 1.0/(delx*delx);
    const float rdy2 = 1.0/(dely*dely);
    const float beta_2 = -omega/(2.0*(rdx2+rdy2));

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i = 1; i <= imax; i++) {
        for (j = 1; j <= jmax; j++) {
            if (flag[i][j] == (C_F | B_NSEW)) {
                p_beta[i][j] = beta_2;
            } else if (flag[i][j] & C_F) {
                p_beta[i][j] = -omega/(
                        (eps_E+eps_W)*rdx2+
                        (eps_N+eps_S)*rdy2
                );
            } else {
                p_beta[i][j] = 0.0f;
            }
        }
    }
}

void splitToRedBlack(float ** const joined, float ** const red, float ** const black,
                     const int imax, const int jmax){
    int i,j;

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i = 0; i < imax+2; i++) {
        for (j = 0; j < jmax+2; j++) {
            if ((i+j) % 2 == 0)
                red[i][j >> 1] = joined[i][j];
            else
                black[i][j >> 1] = joined[i][j];
        }
    }
}

void joinRedBlack(float ** const joined, float ** const red, float ** const black,
                  const int imax, const int jmax) {
    int i,j;

#pragma omp parallel for schedule(static) private(j) default(none)
    for (i = 0; i < imax+2; i++) {
        for (j = 0; j < jmax+2; j++) {
            if ((i+j) % 2 == 0)
                joined[i][j] = red[i][j >> 1];
            else
                joined[i][j] = black[i][j >> 1];
        }
    }
}

/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void updateVelocity(float ** const u, float ** const v, float ** const f, float ** const g, float ** const p,
                    char ** const flag, const int imax, const int jmax, const float del_t, const float delx, const float dely)
{
    int i, j;

    // Loop was fused and parallelized
#pragma omp parallel for schedule(static) private(j) default(none)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }

            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
        }
    }
}


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void setTimestepInterval(float *del_t, int imax, int jmax, float delx,
                         float dely, float **u, float **v, float Re, float tau)
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe;

    // del_t satisfying CFL conditions
    if (tau >= 1.0e-10) { // else no time stepsize control
        umax = 1.0e-10;
        vmax = 1.0e-10;

        // Loop was fused and parallelized
#pragma omp parallel for schedule(static) private(j) shared(imax, jmax, u, v) default(none) reduction(max:umax) reduction(max:vmax)
        for (i=0; i<=imax+1; i++) {
            // TODO - check if j=1 is right here?
            for (j=1; j<=jmax+1; j++) {
                umax = max(fabsf(u[i][j]), umax);
                vmax = max(fabsf(v[i][j]), vmax);
            }
        }

        deltu = delx/umax;
        deltv = dely/vmax;
        // This used to be deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;
        // the original version has 2.0 at the end, but this only ends up doing the rest of the equation, promoting it to double, dividing it, and demoting back to int.
        // this is equivalent to dividing by 2.0f without any double-promotions.
        deltRe = 1.0f/(1.0f/(delx*delx)+1.0f/(dely*dely))*Re/2.0f;

        if (deltu<deltv) {
            *del_t = min(deltu, deltRe);
        } else {
            *del_t = min(deltv, deltRe);
        }
        *del_t = tau * (*del_t); // multiply by safety factor
    }
}

void applyBoundaryConditions(float **u, float **v, char **flag,
                             int imax, int jmax, float ui, float vi)
{
    int i, j;

    for (j=0; j<=jmax+1; j++) {
        // Fluid freely flows in from the west
        u[0][j] = u[1][j];
        v[0][j] = v[1][j];

        // Fluid freely flows out to the east
        u[imax][j] = u[imax-1][j];
        v[imax+1][j] = v[imax][j];
    }

    for (i=0; i<=imax+1; i++) {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v[i][jmax] = 0.0;
        u[i][jmax+1] = u[i][jmax];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
#pragma omp parallel for schedule(static) private(j) shared(u, v, flag, imax, jmax) default(none)
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & B_NSEW) {
                switch (flag[i][j]) {
                    case B_N:
                        v[i][j]   = 0.0;
                        u[i][j]   = -u[i][j+1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_E:
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        v[i][j-1] = -v[i+1][j-1];
                        break;
                    case B_S:
                        v[i][j-1] = 0.0;
                        u[i][j]   = -u[i][j-1];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_W:
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        v[i][j-1] = -v[i-1][j-1];
                        break;
                    case B_NE:
                        v[i][j]   = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j-1] = -v[i+1][j-1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_SE:
                        v[i][j-1] = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_SW:
                        v[i][j-1] = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_NW:
                        v[i][j]   = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j-1] = -v[i-1][j-1];
                        u[i][j]   = -u[i][j+1];
                        break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    v[0][0] = 2*vi-v[1][0];
    for (j=1;j<=jmax;j++) {
        u[0][j] = ui;
        v[0][j] = 2*vi-v[1][j];
    }
}

void calculateFluidmask(int ** fluidmask, const char** flag,
                        int imax, int jmax) {
    int i,j;
    for (i = 0; i < imax+2; i++) {
        for (j = 0; j < jmax+2; j++) {
            fluidmask[i][j] = (flag[i][j] & C_F) ? 0xFFFFFFFF : 0;
        }
    }
}

void splitFluidmaskToSurroundedMask(const int** fluidmask,
                                    int** red, int** black,
                                    const int imax, const int jmax) {
    int i,j;

    for (i = 0; i < imax+2; i++) {
        for (j = 0; j < jmax+2; j++) {
            int val = 0;
            if (i == 0 || i == imax+1 || j == 0 || j == jmax+1) {
                val = 0;
            } else {
                val = fluidmask[i][j] &
                      fluidmask[i+1][j] & fluidmask[i-1][j] &
                      fluidmask[i][j+1] & fluidmask[i][j-1];
            }

            if ((i+j) % 2 == 0)
                red[i][j >> 1] = val;
            else
                black[i][j >> 1] = val;
        }
    }
}

}