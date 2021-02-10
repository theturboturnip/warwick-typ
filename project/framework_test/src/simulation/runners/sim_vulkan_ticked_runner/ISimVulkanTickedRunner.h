//
// Created by samuel on 27/08/2020.
//

#pragma once

#include <memory>
#include <optional>

#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/SimSnapshot.h"
#include "simulation/file_format/FluidParams.h"
#include "simulation/memory/vulkan/VulkanSimulationAllocator.h"
#include <vulkan/vulkan.hpp>
#include <rendering/vulkan/VulkanContext.h>
#include <memory/FrameSetAllocator.h>

class ISimVulkanTickedRunner {
protected:
    ISimVulkanTickedRunner() = default;
public:
    virtual ~ISimVulkanTickedRunner() = default;

    virtual VulkanFrameSetAllocator* prepareBackend(const FluidParams& p, const SimSnapshot& snapshot, size_t frameCount) = 0;
    // Set doSim to false if you just want to signal the semaphores
    virtual void tick(float timeToRun, bool waitOnRender, bool doSim, size_t frameToWriteIdx) = 0;

    static std::unique_ptr<ISimVulkanTickedRunner> getForBackend(SimulationBackendEnum backendType,
                                                                 VulkanContext& context, vk::Semaphore renderFinished, vk::Semaphore simFinished);
};
