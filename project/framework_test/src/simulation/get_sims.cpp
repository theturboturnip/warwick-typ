//
// Created by samuel on 20/06/2020.
//

#include "simulation/backends/null/NullSimulation.h"
#include "util/fatal_error.h"
#include "get_sims.h"

std::unique_ptr<ISimulation> getSimulation(SimulationBackend backend) {
    switch(backend) {
        case Null:
            return std::make_unique<NullSimulation>();
        default:
            FATAL_ERROR("CUDA (enum val %d) isn't defined yet!\n", backend);
    }
    return nullptr;
}