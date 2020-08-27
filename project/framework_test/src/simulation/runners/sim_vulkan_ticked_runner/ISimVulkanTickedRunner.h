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

class ISimVulkanTickedRunner {
protected:
    ISimVulkanTickedRunner() = default;
public:
    virtual ~ISimVulkanTickedRunner() = default;

    virtual VulkanSimulationBuffers prepareBackend(const FluidParams& p, const SimSnapshot& snapshot) = 0;
    virtual void tick(float timeToRun, bool waitOnRender) = 0;

    static std::unique_ptr<ISimVulkanTickedRunner> getForBackend(SimulationBackendEnum backendType,
                                                                 vk::Device device, vk::PhysicalDevice physicalDevice, vk::Semaphore renderFinished, vk::Semaphore simFinished);
};
