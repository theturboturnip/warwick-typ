//
// Created by samuel on 28/06/2020.
//

#include "CpuOptimizedAdaptedSimBackend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>

#include "simulation/backends/original/simulation.h"

CpuOptimizedAdaptedSimBackend::CpuOptimizedAdaptedSimBackend(const FluidParams & params, const SimSnapshot& s) :
    CpuSimBackendBase(params, s),
    p_beta(imax+2, jmax+2, 0),
    p_beta_red(imax+2, (jmax+2)/2, 0),
    p_beta_black(imax+2, (jmax+2)/2, 0),

    p_red(imax+2, (jmax+2)/2, 0),
    p_black(imax+2, (jmax+2)/2, 0),

    rhs_red(imax+2, (jmax+2)/2, 0),
    rhs_black(imax+2, (jmax+2)/2, 0),

    fluidmask(imax+2, jmax+2, 0),
    surroundmask_red(imax+2, (jmax+2)/2, 0),
    surroundmask_black(imax+2, (jmax+2)/2, 0) {
    DASSERT(jmax % 2 == 0);

    OriginalOptimized::calculatePBeta(p_beta.get_pointers(), flag.get_pointers(), imax, jmax, delx, dely, params.poisson_error_threshold, params.poisson_omega);
    OriginalOptimized::splitToRedBlack(p.get_pointers(), p_red.get_pointers(), p_black.get_pointers(), imax, jmax);
    OriginalOptimized::splitToRedBlack(p_beta.get_pointers(), p_beta_red.get_pointers(), p_beta_black.get_pointers(), imax, jmax);
    OriginalOptimized::calculateFluidmask(fluidmask.get_pointers(), (const char **) flag.get_pointers(), imax, jmax);
    OriginalOptimized::splitFluidmaskToSurroundedMask((const int **) (fluidmask.get_pointers()), surroundmask_red.get_pointers(), surroundmask_black.get_pointers(), imax, jmax);
}

void CpuOptimizedAdaptedSimBackend::tick(float del_t) {
    const int ifluid = (imax * jmax) - ibound;

    // Use the <float> variant for slight inaccuracy but GPU bit-parity
    OriginalOptimized::computeTentativeVelocity<float>(u.get_pointers(), v.get_pointers(), f.get_pointers(), g.get_pointers(), flag.get_pointers(),
                                       imax, jmax, del_t, delx, dely, gamma, Re);
    OriginalOptimized::computeRhs(f.get_pointers(), g.get_pointers(), rhs.get_pointers(), flag.get_pointers(),
                         imax, jmax, del_t, delx, dely);

    float res = 0;
    if (ifluid > 0) {
        OriginalOptimized::poissonSolver<false>(p.get_pointers(), p_red.get_pointers(), p_black.get_pointers(),
                                                p_beta.get_pointers(), p_beta_red.get_pointers(), p_beta_black.get_pointers(),
                                                rhs.get_pointers(), rhs_red.get_pointers(), rhs_black.get_pointers(),
                                                fluidmask.get_pointers(), surroundmask_black.get_pointers(),
                                                flag.get_pointers(), imax, jmax,
                                                delx, dely,
                                                params.poisson_error_threshold, params.poisson_max_iterations, params.poisson_omega,
                                                ifluid);
    }

    OriginalOptimized::updateVelocity(u.get_pointers(), v.get_pointers(),
                                      f.get_pointers(), g.get_pointers(),
                                      p.get_pointers(), flag.get_pointers(),
                                      imax, jmax, del_t, delx, dely);
    OriginalOptimized::applyBoundaryConditions(u.get_pointers(), v.get_pointers(), flag.get_pointers(), imax, jmax, params.initial_velocity_x, params.initial_velocity_y);
}

float CpuOptimizedAdaptedSimBackend::findMaxTimestep() {
    float delta_t = -1;
    OriginalOptimized::setTimestepInterval(&delta_t,
                                           imax, jmax,
                                           delx, dely,
                                           u.get_pointers(), v.get_pointers(),
                                           params.Re,
                                           params.timestep_safety
    );
    DASSERT(delta_t != -1);
    return delta_t;
}
