//
// Created by samuel on 27/08/2020.
//

#include "ISimVulkanTickedRunner.h"

#include <util/fatal_error.h>

#if CUDA_ENABLED
#include "simulation/backends/cuda/CudaBackendV1.cuh"
#include "SimVulkanTickedCudaRunner.inl"
#endif

std::unique_ptr<ISimVulkanTickedRunner> ISimVulkanTickedRunner::getForBackend(
        SimulationBackendEnum backendType,
        VulkanContext& context
        ) {
    switch(backendType) {
#if CUDA_ENABLED
        case CUDA:
            return std::make_unique<SimVulkanTickedCudaRunner<CudaBackendV1<false>>>(context);
#endif
        default:
            FATAL_ERROR("Enum val %d doesn't have an ISimVulkanTickedRunner!\n", backendType);
    }
    return nullptr;
}
