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


struct SystemWorkerIn {
    uint32_t simFrameIndex;
    bool shouldSimParticles;

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
    float min=0, max=1;
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

    enum class ScalarQuantity : size_t {
        None=0,
        VelocityX=1,
        VelocityY=2,
        Pressure=3,
        Vorticity=4
    };
    static std::array<const char*, 5> scalarQuantity;
    enum class VectorQuantity : size_t {
        None=0,
        Velocity=1
    };
    static std::array<const char*, 2> vectorQuantity;
    // Stream contour lines with Zeta
    bool overlayStreamlines = false;
    ScalarQuantity vizScalar = ScalarQuantity::Vorticity;
    VizValueRange vizScalarRange;
    VectorQuantity vizVector = VectorQuantity::None;
    VizValueRange vizVectorMagnitudeRange;
    // Particle Options
    bool showParticles = false;
    bool renderParticleGlyphs = true;
    enum class ParticleTrailType : size_t {
        None=0,
        Streakline=1,
        Pathline=2, // < are these different?
        Ribbon=3    // <
    };
    static std::array<const char*, 4> particleTrailType;
    ParticleTrailType trailType = ParticleTrailType::None;
    float trailLength = 0;

    void transferImageLayout(vk::CommandBuffer cmdBuffer,
                             vk::Image image,
                             vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                             vk::AccessFlags oldAccess, vk::AccessFlags newAccess,
                             vk::PipelineStageFlags oldStage = vk::PipelineStageFlagBits::eTopOfPipe,
                             vk::PipelineStageFlags newStage = vk::PipelineStageFlagBits::eTopOfPipe);
    void showRange(VizValueRange* range);

public:
    explicit SystemWorker(VulkanSimAppData& data);

    SystemWorkerOut work(SystemWorkerIn input);
};

#include "rendering/threads/IWorkerThread_Impl.inl"
using SystemWorkerThread = IWorkerThread_Impl<SystemWorker, SystemWorkerIn, SystemWorkerOut>;
#include "rendering/threads/WorkerThreadController.h"
using SystemWorkerThreadController = WorkerThreadController<SystemWorkerIn, SystemWorkerOut>;