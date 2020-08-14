//
// Created by samuel on 20/06/2020.
//

#pragma once

enum SimulationBackendEnum {
    Null,
#if CUDA_ENABLED
    CUDA,
#endif
    CpuSimple,
    CpuOptimized,
    CpuAdapted,
};