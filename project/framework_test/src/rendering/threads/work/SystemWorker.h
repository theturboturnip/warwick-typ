//
// Created by samuel on 22/08/2020.
//

#pragma once

#include "rendering/vulkan/VulkanSimApp.h"
#include <SDL_vulkan.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <rendering/vulkan/helpers/VulkanDeviceMemory.h>
#include <rendering/vulkan/helpers/VulkanBackedFramebuffer.h>
#include <rendering/vulkan/VulkanSimAppData.h>
#include <rendering/vulkan/viz/enums.h>


struct SystemWorkerIn {
    uint32_t simFrameIndex;
    bool shouldSimParticles;
    float thisSimTickLength;
    float lastFrameTime;

    struct PerfData {
        std::array<float, 32> frameTimes;
        uint32_t currentFrameNum;

        std::array<float, 32> simFrameTimes;
        std::array<float, 32> simTickLengths;
        uint32_t simFrameNum;

        double elapsedRealTime;
        double elapsedSimTime;
        double elapsedRealTimeWhileSimWanted;
    } perf;
};

struct SystemWorkerOut {
    bool wantsQuit = false;
    bool wantsRunSim = false;
};

struct VizValueRange {
    // If true, the threshold is set to the minimum/maximum values present.
    // If false, the threshold is using min/max. Any values outside the range are disabled.
    bool autoRange = true;
    float min=-1, max=1;
};

/**
 * Processes SDL input, and builds the command buffer for the next frame
 */
class SystemWorker {
    VulkanSimAppData& data;
    // Shortcut for data.global
    VulkanSimAppData::Global global;


    // internal
    bool showDemoWindow = true;
    bool wantsRunSim = false;

    // Stream contour lines with Zeta
    bool overlayStreamlines = false;
    ScalarQuantity vizScalar = ScalarQuantity::Pressure;
    VizValueRange vizScalarRange;
    VectorQuantity vizVector = VectorQuantity::Velocity;
    VizValueRange vizVectorMagnitudeRange;
    float vizVectorSpacing[2] = {0.5, 0.5};
    float vizVectorSize = 0.04;
    float vizVectorLength = 0.65;
    // Particle Options
    bool simulateParticles = true;
    bool renderParticleGlyphs = true;
    float particleGlyphSize = 0.01;
    bool lockParticleToSimulation = true;
    float particleUnlockedSimFreq = 120;
    float particleSpawnFreq = 10;
    float particleSpawnTimer = 0;
    ParticleTrailType trailType = ParticleTrailType::None;
    float trailLength = 0;

    void transferImageLayout(vk::CommandBuffer cmdBuffer,
                             vk::Image image,
                             vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                             vk::AccessFlags oldAccess, vk::AccessFlags newAccess,
                             vk::PipelineStageFlags oldStage = vk::PipelineStageFlagBits::eTopOfPipe,
                             vk::PipelineStageFlags newStage = vk::PipelineStageFlagBits::eTopOfPipe);
    void fullMemoryBarrier(
        vk::CommandBuffer cmdBuffer,
        vk::PipelineStageFlags oldStage, vk::PipelineStageFlags newStage,
        vk::AccessFlags oldAccess, vk::AccessFlags newAccess
    );
    void showRange(VizValueRange* range);

public:
    explicit SystemWorker(VulkanSimAppData& data);

    SystemWorkerOut work(SystemWorkerIn input);
};

#include "rendering/threads/IWorkerThread_Impl.inl"
using SystemWorkerThread = IWorkerThread_Impl<SystemWorker, SystemWorkerIn, SystemWorkerOut>;
#include "rendering/threads/WorkerThreadController.h"
using SystemWorkerThreadController = WorkerThreadController<SystemWorkerIn, SystemWorkerOut>;