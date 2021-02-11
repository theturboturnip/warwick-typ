//
// Created by samuel on 27/08/2020.
//

#pragma once

#if CUDA_ENABLED
#include "ISimVulkanTickedRunner.h"
#include <simulation/backends/cuda/BaseCudaBackend.cuh>
#include <simulation/backends/cuda/utils/CudaVulkanSemaphore.cuh>
#include <simulation/file_format/FluidParams.h>
#include <memory/FrameSetAllocator.h>


template<typename CudaBackend>
class SimVulkanTickedCudaRunner : public ISimVulkanTickedRunner {
    static_assert(std::is_base_of_v<BaseCudaBackend, CudaBackend>, "SimVulkanTickedCudaRunner must be instantiated with a base class of BaseCudaBackend");

    std::unique_ptr<CudaBackend> backend;
    // TODO - make this allocated in constructor not prepareBackend
    std::unique_ptr<VulkanFrameSetAllocator> allocator;

    VulkanContext& context;
    CudaVulkanSemaphore renderFinishedShouldSim, simFinished;

public:
    using AllocatorType = FrameSetAllocator<MType::VulkanCuda, typename CudaBackend::Frame>;

    SimVulkanTickedCudaRunner(VulkanContext& context, vk::Semaphore renderFinishedShouldSim, vk::Semaphore simFinished)
        : context(context),
          renderFinishedShouldSim(*context.device, renderFinishedShouldSim),
          simFinished(*context.device, simFinished)
    {}

     VulkanFrameSetAllocator* prepareBackend(const FluidParams& p, const SimSnapshot& snapshot, size_t frameCount) override {
        auto specificAllocator = std::make_unique<AllocatorType>(context, snapshot.simSize.padded_pixel_size, frameCount);
        backend = std::make_unique<CudaBackend>(specificAllocator->frames, p, snapshot);

        allocator = std::move(specificAllocator);

        return allocator.get();
    }

    void tick(float timeToRun, bool waitOnRender, bool doSim, size_t frameToWriteIdx) override {
        if (waitOnRender) {
            //fprintf(stderr, "Waiting on renderFinishedShouldSim\n");
            renderFinishedShouldSim.waitForAsync(backend->stream);
        }

        float currentTime = 0;
        while(doSim && currentTime < timeToRun) {
            //fprintf(stderr, "Starting findMaxTimestep\n");
            float maxTimestep = backend->findMaxTimestep();
            //fprintf(stderr, "Ended findMaxTimestep\n");
            if (currentTime + maxTimestep > timeToRun)
                maxTimestep = timeToRun - currentTime;
            //fprintf(stderr, "Starting backend->tick()\n");
            backend->tick(maxTimestep, frameToWriteIdx);
            //fprintf(stderr, "Finishing findMaxTimestep\n");
            currentTime += maxTimestep;
        }
        //fprintf(stderr, "Signalling simFinished\n");
        simFinished.signalAsync(backend->stream);
        CHECKED_CUDA(cudaStreamSynchronize(backend->stream));
    }
};
#endif