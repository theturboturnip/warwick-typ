//
// Created by samuel on 30/06/2020.
//

#include "simulation/SimulationBackendEnum.h"

#include "ISimFixedTimeRunner.h"
#include "SimFixedTimeRunner.inl"

#include "simulation/backends/cpu_simple/CpuOptimizedSimBackend.h"
#include "simulation/backends/cpu_simple/CpuSimpleSimBackend.h"
#include "simulation/backends/cpu_simple/optimized/CpuOptimizedAdaptedSimBackend.h"
#include "simulation/backends/null/NullSimulation.h"
#if CUDA_ENABLED
#include "simulation/backends/cuda/CudaBackendV1.cuh"
#endif

#include "util/fatal_error.h"

std::unique_ptr<ISimFixedTimeRunner> ISimFixedTimeRunner::getForBackend(SimulationBackendEnum backendType) {
    switch(backendType) {
        case Null:
            return std::make_unique<SimFixedTimeRunner<NullSimulation, MType::Cpu>>();
        case CpuSimple:
            return std::make_unique<SimFixedTimeRunner<CpuSimpleSimBackend, MType::Cpu>>();
        case CpuOptimized:
            return std::make_unique<SimFixedTimeRunner<CpuOptimizedSimBackend, MType::Cpu>>();
        case CpuAdapted:
            return std::make_unique<SimFixedTimeRunner<CpuOptimizedAdaptedSimBackend, MType::Cpu>>();
#if CUDA_ENABLED
        case CUDA:
            return std::make_unique<SimFixedTimeRunner<CudaBackendV1<true>, MType::Cuda>>();
#endif
        default:
            FATAL_ERROR("Enum val %d doesn't have an ISimFixedTimeRunner!\n", backendType);
    }
    return nullptr;
}