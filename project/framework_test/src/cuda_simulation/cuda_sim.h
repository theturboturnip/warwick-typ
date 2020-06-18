//
// Created by samuel on 18/06/2020.
//

#pragma once

#include "cuda_memory_wrappers.h"

class Simulation {
public:
    Simulation();
    ~Simulation();

    struct TickData {

    };

    TickData tickSimulationBlocking();

private:
    CUDAUnified2DArray<float> velocity_x;
    CUDAUnified2DArray<float> velocity_y;

    CUDAUnified2DArray<float> tentative_velocity_x;
    CUDAUnified2DArray<float> tentative_velocity_y;

    CUDAUnified2DArray<float> pressure_eq_rhs;
    CUDAUnified2DArray<float> pressure_eq_beta;


};
