//
// Created by samuel on 27/08/2020.
//

#pragma once

#if CUDA_ENABLED
#include "ISimVulkanTickedRunner.h"
#include <simulation/backends/cuda/BaseCudaBackend.cuh>
#include <simulation/backends/cuda/utils/CudaVulkanSemaphore.cuh>
#include <simulation/file_format/FluidParams.h>
#include <simulation/memory/vulkan/CudaVulkan2DAllocator.cuh>
#include <simulation/memory/vulkan/VulkanSimulationAllocator.h>


template<typename CudaBackend>
class SimVulkanTickedCudaRunner : public ISimVulkanTickedRunner {
    static_assert(std::is_base_of_v<BaseCudaBackend, CudaBackend>, "SimVulkanTickedCudaRunner must be instantiated with a base class of BaseCudaBackend");

    std::unique_ptr<VulkanSimulationAllocator<CudaVulkan2DAllocator>> allocator;
    std::unique_ptr<CudaBackend> backend;

    CudaVulkanSemaphore hasImage, simFinished;

public:
    SimVulkanTickedCudaRunner(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Semaphore hasImage, vk::Semaphore simFinished)
        : allocator(std::make_unique<VulkanSimulationAllocator<CudaVulkan2DAllocator>>(device, physicalDevice)),
          hasImage(device, hasImage),
          simFinished(device, simFinished)
    {}

    VulkanSimulationBuffers prepareBackend(const FluidParams& p, const SimSnapshot& snapshot) override {
        auto vulkanAllocs = allocator->makeAllocs(snapshot);
        backend = std::make_unique<CudaBackend>(vulkanAllocs.simAllocs, p, snapshot);

        return vulkanAllocs;
    }

    void tick(float timeToRun, bool waitOnRender) override {
        if (waitOnRender) {
            //fprintf(stderr, "Waiting on hasImage\n");
            hasImage.waitForAsync(backend->stream);
        }

        float currentTime = 0;
        while(currentTime < timeToRun) {
            //fprintf(stderr, "Starting findMaxTimestep\n");
            float maxTimestep = backend->findMaxTimestep();
            //fprintf(stderr, "Ended findMaxTimestep\n");
            if (currentTime + maxTimestep > timeToRun)
                maxTimestep = timeToRun - currentTime;
            //fprintf(stderr, "Starting backend->tick()\n");
            backend->tick(maxTimestep);
            //fprintf(stderr, "Finishing findMaxTimestep\n");
            currentTime += maxTimestep;
        }
        //fprintf(stderr, "Signalling simFinished\n");
        simFinished.signalAsync(backend->stream);
        //CHECKED_CUDA(cudaStreamSynchronize(backend->stream));
    }
};
#endif