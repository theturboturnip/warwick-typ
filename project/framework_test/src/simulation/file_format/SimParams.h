//
// Created by samuel on 09/08/2020.
//

#pragma once

#include <nlohmann/json_fwd.hpp>

#include "util/Size.h"

struct SimParams {
    // Size of the simulation grid in pixels
    Size<int> pixel_size;
    // Size of the simulation grid in meters
    Size<float> physical_size;

    // Meters/pixel for x and y
    [[nodiscard]] inline float del_x() const {
        return physical_size.x/pixel_size.x;
    }
    [[nodiscard]] inline float del_y() const {
        return physical_size.y/pixel_size.y;
    }

    struct {
        float Re; // Reynolds Number - 150 in old sim
    } fluid;

    struct {
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
        // If you use central differences, the "discretization error" is O(dx^2),
        // but if dx is too large then the simulation can begin to oscillate. (p23)
        // If you use forward/backwards differences this is completely prevented,
        // but the discretization error drops to O(dx).
        // Combining these two using a weighted average allows for a compromise between
        // a stable simulation and a low discretization error.
        // gamma is the weight of that average
        //      gamma * upwind diff + (1-gamma) * central diff
        // Note - on p30 it states gamma should be chosen such that
        // gamma >= max(udt/dx, vdt/dy)
        // This works because dt is chosen such that vdt/dy < 1, udt/dx < 1.
        // If you take a constant gamma like we do, it could be important to factor that into dt choice - this is covered by tau=0.5 I think?
        float gamma; // Upwind differencing factor in PDE discretization - (0.9) in old sim

        struct {
            // The pressure calculation is performed using Successive OverRelaxation (SOR)
            // This is an iterative algorithm.

            // Maximum iteration count per tick.
            int max_iterations;
            // Stopping error threshold - once the error is below this point, the iterations stop as the system is determined to be stable.
            float error_threshold;
            // Relaxation parameter for SOR - between [0, 2] and usually chosen as 1.7.
            float omega;
        } redblack_poisson;
    } sim;
};

void to_json(nlohmann::json& j, const SimParams& p);
void from_json(const nlohmann::json& j, SimParams& p);