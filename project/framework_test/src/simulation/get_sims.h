//
// Created by samuel on 20/06/2020.
//

#pragma once

#include <memory>

#include "interface.h"

enum SimulationBackend {
    Null,
    CUDA,
};

std::unique_ptr<ISimulation> getSimulation(SimulationBackend backend);