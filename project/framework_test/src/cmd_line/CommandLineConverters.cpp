//
// Created by samuel on 02/07/2020.
//

#include "CommandLineConverters.h"

template<typename X, typename Y>
std::map<Y,X> flip_map(std::map<X, Y> map) {
    auto newMap = std::map<Y,X>();
    for (const auto& kvp: map) {
        newMap[kvp.second] = kvp.first;
    }
    return newMap;
}

CommandLineConverters::CommandLineConverters()
    : backendMap({
                 {"null", SimulationBackendEnum::Null},
                 {"cpu_simple", SimulationBackendEnum::CpuSimple},
                 {"cpu_fast", SimulationBackendEnum::CpuOptimized},
#if CUDA_ENABLED
                 {"cuda", SimulationBackendEnum::CUDA}
#endif
                }),
      backendToStrMap(flip_map(backendMap)),
// TODO: Once CUDA is implemented, remove this "false &&"
#if false && CUDA_ENABLED
      defaultBackend(SimulationBackendEnum::CUDA)
#else
      defaultBackend(SimulationBackendEnum::CpuOptimized)
#endif
{}
