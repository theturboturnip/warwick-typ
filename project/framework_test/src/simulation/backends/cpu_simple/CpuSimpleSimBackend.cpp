//
// Created by samuel on 22/06/2020.
//

#include "CpuSimpleSimBackend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

// Copy of constants.h
/* Macros for poisson(), denoting whether there is an obstacle cell
 * adjacent to some direction
 */
#define eps_E ((flag[i+1][j] & C_F)?1:0)
#define eps_W ((flag[i-1][j] & C_F)?1:0)
#define eps_N ((flag[i][j+1] & C_F)?1:0)
#define eps_S ((flag[i][j-1] & C_F)?1:0)

CpuSimpleSimBackend::CpuSimpleSimBackend(std::vector<BaseFrame> frames, const FluidParams& params, const SimSnapshot& s) :
    CpuSimBackendBase(params, s),
    frames(std::move(frames)),

    u(frames[0].u.as_cpu()),
    v(frames[0].v.as_cpu()),
    f(frames[0].f.as_cpu()),
    g(frames[0].g.as_cpu()),
    p(frames[0].p.as_cpu()),
    rhs(frames[0].rhs.as_cpu()),
    flag(frames[0].flag.as_cpu())
{
    DASSERT(frames.size() == 1);

    BaseFrame& frame = frames[0];

    frame.u.memcpy_in(s.velocity_x);
    frame.v.memcpy_in(s.velocity_y);
    frame.f.zero_out();
    frame.g.zero_out();
    frame.p.memcpy_in(s.pressure);
    frame.rhs.zero_out();
    frame.flag.memcpy_in(s.get_legacy_cell_flags());
}

float CpuSimpleSimBackend::findMaxTimestep() {
    int i, j;
    float umax, vmax, deltu, deltv, deltRe;

    //    /* del_t satisfying CFL conditions */
    //    if (tau >= 1.0e-10) { /* else no time stepsize control */
    umax = 1.0e-10;
    vmax = 1.0e-10;
    for (i = 0; i <= imax + 1; i++) {
        for (j = 1; j <= jmax + 1; j++) {
            umax = max(fabs(u[i][j]), umax);
        }
    }
    for (i = 1; i <= imax + 1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
            vmax = max(fabs(v[i][j]), vmax);
        }
    }

    deltu = delx / umax;
    deltv = dely / vmax;
    deltRe = 1 / (1 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

    float del_t;
    if (deltu < deltv) {
        del_t = min(deltu, deltRe);
    } else {
        del_t = min(deltv, deltRe);
    }
    del_t = tau * (del_t); /* multiply by safety factor */
    return del_t;
//}
}

int CpuSimpleSimBackend::tick(float del_t) {
    const int ifluid = (imax * jmax) - ibound;

    computeTentativeVelocity(del_t);
    computeRhs(del_t);

    float res = 0.0f;
    if (ifluid > 0) {
        poissonSolver(&res, ifluid);
    }

    updateVelocity(del_t);
    applyBoundaryConditions();

    return 0;
}

// Computation of tentative velocity field (f, g)
void CpuSimpleSimBackend::computeTentativeVelocity(float del_t )
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                         gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                         (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                         gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                        /(4.0*delx);
                duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                         gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                         (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                         gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                        /(4.0*dely);
                laplu = (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx+
                        (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely;

                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                         gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                         (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                         gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                        /(4.0*delx);
                dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                         gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                         (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                         gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                        /(4.0*dely);

                laplv = (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx+
                        (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely;

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


// Calculate the right hand side of the pressure equation
void CpuSimpleSimBackend::computeRhs(float del_t)
{
    int i, j;

    for (i=1;i<=imax;i++) {
        for (j=1;j<=jmax;j++) {
            if (flag[i][j] & C_F) {
                // only for fluid and non-surface cells
                rhs[i][j] = (
                                    (f[i][j]-f[i-1][j])/delx +
                                    (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
            }
        }
    }
}


// Red/Black SOR to solve the poisson equation
int CpuSimpleSimBackend::poissonSolver(float *res, int ifull)
{
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;

    int rb; // Red-black value.

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    // Calculate sum of squares
    for (i = 1; i <= imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }

    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }

    // Red/Black SOR-iteration

    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
            for (i = 1; i <= imax; i++) {
                for (j = 1; j <= jmax; j++) {
                    if ((i+j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        p[i][j] = (1.-omega)*p[i][j] -
                                  beta_2*(
                                          (p[i+1][j]+p[i-1][j])*rdx2
                                          + (p[i][j+1]+p[i][j-1])*rdy2
                                          -  rhs[i][j]
                                  );
                    } else if (flag[i][j] & C_F) {
                        // modified star near boundary
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p[i][j] = (1.-omega)*p[i][j] -
                                  beta_mod*(
                                          (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                                          + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                                          - rhs[i][j]
                                  );
                    }
                } // end of j
            } // end of i

        } // end of rb
        // Partial computation of residual
        *res = 0.0;
        for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    // only fluid cells
                    add = (eps_E*(p[i+1][j]-p[i][j]) -
                           eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                          (eps_N*(p[i][j+1]-p[i][j]) -
                           eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    *res += add*add;
                }
            }
        }
        *res = sqrt((*res)/ifull)/p0;

        // convergence?
        if (*res<eps) break;
    } // end of iter

    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void CpuSimpleSimBackend::updateVelocity(float del_t)
{
    int i, j;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            // only if both adjacent cells are fluid cells
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
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
 * TODO: On the GPU this could be changed to do half-step, quarter-step etc. which would make it simpler to get out consistent units
 */
/*void CpuSimpleSimBackend::setTimestepInterval()
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe;

    // del_t satisfying CFL conditions
    if (tau >= 1.0e-10) { // else no time stepsize control
        umax = 1.0e-10;
        vmax = 1.0e-10;
        for (i=0; i<=imax+1; i++) {
            for (j=1; j<=jmax+1; j++) {
                umax = max(fabs(u[i][j]), umax);
            }
        }
        for (i=1; i<=imax+1; i++) {
            for (j=0; j<=jmax+1; j++) {
                vmax = max(fabs(v[i][j]), vmax);
            }
        }

        deltu = delx/umax;
        deltv = dely/vmax;
        deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

        if (deltu<deltv) {
            del_t = min(deltu, deltRe);
        } else {
            del_t = min(deltv, deltRe);
        }
        del_t = tau * (del_t); // multiply by safety factor
    }
}*/

void CpuSimpleSimBackend::applyBoundaryConditions()
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

SimSnapshot CpuSimpleSimBackend::get_snapshot() {
    BaseFrame &frame = frames[0];

    auto snap = SimSnapshot(simSize);
    snap.velocity_x = frame.u.extract_data();
    snap.velocity_y = frame.v.extract_data();
    snap.pressure = frame.p.extract_data();
    snap.cell_type = SimSnapshot::cell_type_from_legacy(frame.flag.extract_data());
    return snap;
}
LegacySimDump CpuSimpleSimBackend::dumpStateAsLegacy() {
    return get_snapshot().to_legacy();
}

