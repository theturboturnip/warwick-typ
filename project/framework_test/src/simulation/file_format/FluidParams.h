//
// Created by samuel on 09/08/2020.
//

#pragma once

#include <nlohmann/json_fwd.hpp>

#include "LegacySimSize.h"
#include "util/Size.h"

struct FluidParams {
    FluidParams() = default;
    static FluidParams make_aca_default();

    static FluidParams from_file(std::string path);
    void to_file(std::string path) const;

    float Re; // Reynolds Number - 150 in old sim

    float initial_velocity_x;
    float initial_velocity_y;

    // The base timestep that is subdivided by the runner
    // i.e. if timestep_divisor = 60, each timestep will be at most 1/60th of a second,
    // and could be 1/120th, 1/240th etc.
    int timestep_divisor;
    // The maximum timestep divisor.
    // if max_timestep_divisor = 240, the minimum timestep for the simulation will be 1/240th of a second.
    int max_timestep_divisor;

    // Safety factor for timestep control - (0.5) in old sim
    float timestep_safety;

    // When solving second-order differential equations numerically,
    // you can approximate the derivitives either with central differences
    //      (dui/dx = (u(xi+1) - u(xi-1))/2dx)
    // or with forward/backward differences.
    //      (dui/dx = (u(xi+1) - u(xi))/dx for forward difference)
    // If you use central differences, the "discretization error" is big-O(dx^2),
    // but if dx is too large then the simulation can begin to oscillate. (p23)
    // If you use forward/backwards differences this is completely prevented,
    // but the discretization error drops to big-O(dx).
    // Combining these two using a weighted average allows for a compromise between
    // a stable simulation and a low discretization error.
    // gamma is the weight of that average
    //      gamma * upwind diff + (1-gamma) * central diff
    // Note - on p30 it states gamma should be chosen such that
    // gamma >= max(udt/dx, vdt/dy)
    // This works because dt is chosen such that vdt/dy < 1, udt/dx < 1.
    // If you take a constant gamma like we do, it could be important to factor that into dt choice - this is covered by tau=0.5 I think?
    float gamma; // Upwind differencing factor in PDE discretization - (0.9) in old sim

    // The pressure calculation is performed using Successive OverRelaxation (SOR)
    // This is an iterative algorithm.

    // Maximum iteration count per tick.
    int poisson_max_iterations;
    // Stopping error threshold - once the error is below this point, the iterations stop as the system is determined to be stable.
    float poisson_error_threshold;
    // Relaxation parameter for SOR - between [0, 2] and usually chosen as 1.7.
    float poisson_omega;
};

void to_json(nlohmann::ordered_json& j, const FluidParams & p);
void from_json(const nlohmann::ordered_json& j, FluidParams & p);