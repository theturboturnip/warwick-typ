//
// Created by samuel on 20/06/2020.
//

#include "get_sims.h"

#include "simulation/backends/cpu_simple/CpuSimpleSimulation.h"
#include "simulation/backends/null/NullSimulation.h"
#include "util/fatal_error.h"

std::unique_ptr<ISimulation> getSimulation(SimulationBackend backend) {
    switch(backend) {
        case Null:
            return std::make_unique<NullSimulation>();
        case CpuSimple:
            return std::make_unique<CpuSimpleSimulation>();
        default:
            FATAL_ERROR("CUDA (enum val %d) isn't defined yet!\n", backend);
    }
    return nullptr;
}