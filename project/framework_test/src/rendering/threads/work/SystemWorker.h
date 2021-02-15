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
    uint32_t swapchainImageIndex;
    uint32_t simFrameIndex;

    struct PerfData {
        std::array<float, 32> frameTimes;
        uint32_t currentFrameNum;

        std::array<float, 32> simFrameTimes;
        std::array<float, 32> simTickLengths;
        uint32_t simFrameNum;

        double elapsedRealTime;
        double elapsedSimTime;
        double elapsedRealTimeDuringSim;
    } perf;
};

struct SystemWorkerOut {
    bool wantsQuit = false;
    bool wantsRunSim = false;
    vk::CommandBuffer graphicsCmdBuffer;
    vk::CommandBuffer computeCmdBuffer;
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

    // In case the constants change over time i.e. for color
    VulkanSimPipelineSet::SimFragPushConstants simBuffersPushConstants;

    void transferImageLayout(vk::CommandBuffer cmdBuffer,
                             vk::Image image,
                             vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                             vk::AccessFlags oldAccess, vk::AccessFlags newAccess,
                             vk::PipelineStageFlags oldStage = vk::PipelineStageFlagBits::eTopOfPipe,
                             vk::PipelineStageFlags newStage = vk::PipelineStageFlagBits::eTopOfPipe);

public:
    explicit SystemWorker(VulkanSimAppData& data);

    SystemWorkerOut work(SystemWorkerIn input);
};

#include "rendering/threads/IWorkerThread_Impl.inl"
using SystemWorkerThread = IWorkerThread_Impl<SystemWorker, SystemWorkerIn, SystemWorkerOut>;
#include "rendering/threads/WorkerThreadController.h"
using SystemWorkerThreadController = WorkerThreadController<SystemWorkerIn, SystemWorkerOut>;