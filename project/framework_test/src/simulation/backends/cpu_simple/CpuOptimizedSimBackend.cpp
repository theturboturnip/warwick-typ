//
// Created by samuel on 28/06/2020.
//

#include "CpuOptimizedSimBackend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>

#include "simulation/backends/original/simulation.h"

CpuOptimizedSimBackend::CpuOptimizedSimBackend(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& s) :
    CpuSimBackendBase(params, s),
    frames(std::move(frames)),
    lastWrittenFrame(0)
    {
    DASSERT(jmax % 2 == 0);

    for (auto& frame : frames) {
        resetFrame(frame, s);
    }
}

void CpuOptimizedSimBackend::tick(float del_t, int frameToWriteIdx) {
    const int ifluid = (imax * jmax) - ibound;

    const Frame& previousFrame = frames[lastWrittenFrame];
    Frame& frame = frames[frameToWriteIdx];

    // Use the <double> variant for bit-accuracy
    OriginalOptimized::computeTentativeVelocity<double>(previousFrame.u.as_cpu(), previousFrame.v.as_cpu(), frame.f.as_cpu(), frame.g.as_cpu(), frame.flag.as_cpu(),
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

    lastWrittenFrame = frameToWriteIdx;
}

float CpuOptimizedSimBackend::findMaxTimestep() {
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

void CpuOptimizedSimBackend::resetFrame(CpuOptimizedSimBackend::Frame &frame, const SimSnapshot &s) {
    frame.u.memcpy_in(s.velocity_x);
    frame.v.memcpy_in(s.velocity_y);
    frame.f.zero_out();
    frame.g.zero_out();
    frame.p.memcpy_in(s.pressure);
    frame.rhs.zero_out();
    frame.flag.memcpy_in(s.get_legacy_cell_flags());

    OriginalOptimized::calculatePBeta(frame.p_beta.as_cpu(), frame.flag.as_cpu(), imax, jmax, delx, dely, params.poisson_error_threshold, params.poisson_omega);
    OriginalOptimized::splitToRedBlack(frame.p.as_cpu(), frame.p_red.as_cpu(), frame.p_black.as_cpu(), imax, jmax);
    OriginalOptimized::splitToRedBlack(frame.p_beta.as_cpu(), frame.p_beta_red.as_cpu(), frame.p_beta_black.as_cpu(), imax, jmax);
    OriginalOptimized::calculateFluidmask(frame.fluidmask.as_cpu(), (const char **) frame.flag.as_cpu(), imax, jmax);
    OriginalOptimized::splitFluidmaskToSurroundedMask((const int **) (frame.fluidmask.as_cpu()), frame.surroundmask_red.as_cpu(), frame.surroundmask_black.as_cpu(), imax, jmax);
}

SimSnapshot CpuOptimizedSimBackend::get_snapshot() {
    Frame &frame = frames[lastWrittenFrame];

    auto snap = SimSnapshot(simSize);
    snap.velocity_x = frame.u.extract_data();
    snap.velocity_y = frame.v.extract_data();
    snap.pressure = frame.p.extract_data();
    snap.cell_type = SimSnapshot::cell_type_from_legacy(frame.flag.extract_data());
    return snap;
}
LegacySimDump CpuOptimizedSimBackend::dumpStateAsLegacy() {
    return get_snapshot().to_legacy();
}

CpuOptimizedSimBackend::Frame::Frame(FrameAllocator<MType::Cpu> &alloc, Size<uint32_t> paddedSize) 
: BaseFrame(alloc, paddedSize),
    redBlackSize(paddedSize.x, paddedSize.y/2),

    p_beta(alloc, paddedSize),
    p_beta_red(alloc, redBlackSize),
    p_beta_black(alloc, redBlackSize),
    
    p_red(alloc, redBlackSize),
    p_black(alloc, redBlackSize),
    
    rhs_red(alloc, redBlackSize),
    rhs_black(alloc, redBlackSize),
    
    fluidmask(alloc, paddedSize),
    surroundmask_red(alloc, redBlackSize),
    surroundmask_black(alloc, redBlackSize)
{}
