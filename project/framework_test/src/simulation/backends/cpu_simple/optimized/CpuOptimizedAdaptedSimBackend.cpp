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

CpuOptimizedAdaptedSimBackend::CpuOptimizedAdaptedSimBackend(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& s) :
    CpuOptimizedSimBackend(std::move(frames), params, s){
    DASSERT(jmax % 2 == 0);

}

int CpuOptimizedAdaptedSimBackend::tick(float del_t) {
    const int ifluid = (imax * jmax) - ibound;

    const int nextFrameIdx = (lastWrittenFrame + 1) % frames.size();

    const Frame& previousFrame = frames[lastWrittenFrame];
    Frame& frame = frames[nextFrameIdx];

    // Use the <float> variant for slight inaccuracy but GPU bit-parity
    OriginalOptimized::computeTentativeVelocity<float>(previousFrame.u.as_cpu(), previousFrame.v.as_cpu(), frame.f.as_cpu(), frame.g.as_cpu(), frame.flag.as_cpu(),
                                       imax, jmax, del_t, delx, dely, gamma, Re);
    OriginalOptimized::computeRhs(frame.f.as_cpu(), frame.g.as_cpu(), frame.rhs.as_cpu(), frame.flag.as_cpu(),
                         imax, jmax, del_t, delx, dely);

    float res = 0;
    if (ifluid > 0) {
        OriginalOptimized::poissonSolver<false>(frame.p.as_cpu(), frame.p_red.as_cpu(), frame.p_black.as_cpu(),
                                                frame.p_beta.as_cpu(), frame.p_beta_red.as_cpu(), frame.p_beta_black.as_cpu(),
                                                frame.rhs.as_cpu(), frame.rhs_red.as_cpu(), frame.rhs_black.as_cpu(),
                                                frame.fluidmask.as_cpu(), frame.surroundmask_black.as_cpu(),
                                                frame.flag.as_cpu(), imax, jmax,
                                                delx, dely,
                                                params.poisson_error_threshold, params.poisson_max_iterations, params.poisson_omega,
                                                ifluid);
    }

    OriginalOptimized::updateVelocity(frame.u.as_cpu(), frame.v.as_cpu(),
                                      frame.f.as_cpu(), frame.g.as_cpu(),
                                      frame.p.as_cpu(), frame.flag.as_cpu(),
                                      imax, jmax, del_t, delx, dely);
    OriginalOptimized::applyBoundaryConditions(frame.u.as_cpu(), frame.v.as_cpu(), frame.flag.as_cpu(), imax, jmax, params.initial_velocity_x, params.initial_velocity_y);

    lastWrittenFrame = nextFrameIdx;
    return lastWrittenFrame;
}

float CpuOptimizedAdaptedSimBackend::findMaxTimestep() {
    Frame& frame = frames[lastWrittenFrame];

    float delta_t = -1;
    OriginalOptimized::setTimestepInterval(&delta_t,
                                           imax, jmax,
                                           delx, dely,
                                           frame.u.as_cpu(), frame.v.as_cpu(),
                                           params.Re,
                                           params.timestep_safety
    );
    DASSERT(delta_t != -1);
    return delta_t;
}
