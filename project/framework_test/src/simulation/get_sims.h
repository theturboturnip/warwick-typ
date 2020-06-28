//
// Created by samuel on 20/06/2020.
//

#pragma once

#include <memory>

#include "simulation/runners/ISimTickedRunner.h"

enum SimulationBackend {
    Null,
    CUDA,
    CpuSimple,
    CpuOptimized,
};

std::unique_ptr<ISimTickedRunner> getSimulation(SimulationBackend backend);