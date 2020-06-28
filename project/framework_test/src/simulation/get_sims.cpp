//
// Created by samuel on 20/06/2020.
//

#include "get_sims.h"

#include "simulation/runners/ISimTickedRunner.h"
#include "simulation/backends/cpu_simple/CpuSimpleSimBackend.h"
#include "simulation/backends/cpu_simple/CpuOptimizedSimBackend.h"
#include "simulation/backends/null/NullSimulation.h"
#include "util/fatal_error.h"

#include "simulation/runners/SimTickedRunner.inl"

std::unique_ptr<ISimTickedRunner> getSimulation(SimulationBackend backend) {
    switch(backend) {
        case Null:
            return std::make_unique<SimTickedRunner<NullSimulation>>();
        case CpuSimple:
            return std::make_unique<SimTickedRunner<CpuSimpleSimBackend>>();
        case CpuOptimized:
            return std::make_unique<SimTickedRunner<CpuOptimizedSimBackend>>();
        default:
            FATAL_ERROR("Enum val %d isn't defined yet!\n", backend);
    }
    return nullptr;
}