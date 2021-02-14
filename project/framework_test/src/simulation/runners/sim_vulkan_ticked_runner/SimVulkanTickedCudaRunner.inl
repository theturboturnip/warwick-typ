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
#include <rendering/vulkan/VulkanSimAppData.h>


template<typename CudaBackend>
class SimVulkanTickedCudaRunner : public ISimVulkanTickedRunner {
    static_assert(std::is_base_of_v<BaseCudaBackend, CudaBackend>, "SimVulkanTickedCudaRunner must be instantiated with a base class of BaseCudaBackend");

    std::unique_ptr<CudaBackend> backend;
    std::unique_ptr<VulkanFrameSetAllocator> allocator;

    VulkanContext& context;
    struct Sync {
        CudaVulkanSemaphore renderFinished;
        CudaVulkanSemaphore simFinished;

        Sync(vk::Device device, const VulkanSimAppData::PerFrameData& data)
            : renderFinished(device, *data.computeFinishedShouldSim),
                simFinished(device, *data.simFinished) {}
    };
    std::vector<Sync> frameSemaphores;

public:
    using AllocatorType = FrameSetAllocator<MType::VulkanCuda, typename CudaBackend::Frame>;

    explicit SimVulkanTickedCudaRunner(VulkanContext& context)
        : context(context)
    {}

    VulkanFrameSetAllocator* prepareBackend(const FluidParams& p, const SimSnapshot& snapshot, size_t frameCount) override {
        auto specificAllocator = std::make_unique<AllocatorType>(context, snapshot.simSize.padded_pixel_size,
                                                                 frameCount);
        backend = std::make_unique<CudaBackend>(specificAllocator->frames, p, snapshot);

        allocator = std::move(specificAllocator);

        return allocator.get();
    }

    void prepareSemaphores(VulkanSimAppData& data) override {
        frameSemaphores.clear();
        frameSemaphores.reserve(data.frameData.size());
        for (const auto& frame : data.frameData) {
            frameSemaphores.emplace_back(*context.device, frame);
        }
    }

    void tick(float timeToRun, bool waitForRender, bool doSim, size_t frameIdx) override {
        auto& semaphores = frameSemaphores[frameIdx];
        if (waitForRender) {
//            fprintf(stderr, "Waiting on renderFinishedShouldSim\n");
            semaphores.renderFinished.waitForAsync(backend->stream);
        }

        if (doSim) {
            float currentTime = 0;
            while (currentTime < timeToRun) {
//            fprintf(stderr, "Starting findMaxTimestep\n");
                float maxTimestep = backend->findMaxTimestep();
//            fprintf(stderr, "Ended findMaxTimestep\n");
                if (currentTime + maxTimestep > timeToRun)
                    maxTimestep = timeToRun - currentTime;
//            fprintf(stderr, "Starting backend->tick()\n");
                backend->tick(maxTimestep, frameIdx);
//            fprintf(stderr, "Finishing findMaxTimestep\n");
                currentTime += maxTimestep;
            }
        } else {
            // TODO - copy data from frame A to frame B - requires change to runner
        }
//        fprintf(stderr, "Signalling simFinished\n");
        semaphores.simFinished.signalAsync(backend->stream);
//        CHECKED_CUDA(cudaStreamSynchronize(backend->stream));
    }
};
#endif