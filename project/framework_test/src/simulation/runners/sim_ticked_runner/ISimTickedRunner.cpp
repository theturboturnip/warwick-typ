//
// Created by samuel on 20/06/2020.
//

#include "simulation/SimulationBackendEnum.h"

#include "ISimTickedRunner.h"
#include "SimTickedRunner.inl"

#include "simulation/backends/cpu_simple/CpuSimpleSimBackend.h"
#include "simulation/backends/cpu_simple/CpuOptimizedSimBackend.h"
#include "simulation/backends/null/NullSimulation.h"
#include "util/fatal_error.h"


std::unique_ptr<ISimTickedRunner> ISimTickedRunner::getForBackend(SimulationBackendEnum backendType) {
    switch(backendType) {
        case Null:
            return std::make_unique<SimTickedRunner<NullSimulation>>();
        case CpuSimple:
            return std::make_unique<SimTickedRunner<CpuSimpleSimBackend>>();
        case CpuOptimized:
            return std::make_unique<SimTickedRunner<CpuOptimizedSimBackend>>();
        default:
            FATAL_ERROR("Enum val %d doesn't have an ISimTickedRunner!\n", backendType);
    }
    return nullptr;
}