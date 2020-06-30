//
// Created by samuel on 30/06/2020.
//

#include "simulation/SimulationBackendEnum.h"

#include "ISim10sRunner.h"
#include "Sim10sRunner.inl"

#include "simulation/backends/cpu_simple/CpuSimpleSimBackend.h"
#include "simulation/backends/cpu_simple/CpuOptimizedSimBackend.h"
#include "simulation/backends/null/NullSimulation.h"
#include "util/fatal_error.h"

std::unique_ptr<ISim10sRunner> ISim10sRunner::getForBackend(SimulationBackendEnum backendType) {
    switch(backendType) {
        case Null:
            return std::make_unique<Sim10sRunner<NullSimulation>>();
        case CpuSimple:
            return std::make_unique<Sim10sRunner<CpuSimpleSimBackend>>();
        case CpuOptimized:
            return std::make_unique<Sim10sRunner<CpuOptimizedSimBackend>>();
        default:
            FATAL_ERROR("Enum val %d doesn't have an ISim10sRunner!\n", backendType);
    }
    return nullptr;
}