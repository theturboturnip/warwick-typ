//
// Created by samuel on 22/08/2020.
//
#include "VulkanSimApp.h"

#include "rendering/vulkan/helpers/VulkanDebug.h"

#if CUDA_ENABLED
#include <simulation/backends/cuda/CudaBackendV1.cuh>
#include <simulation/memory/vulkan/CudaVulkan2DAllocator.cuh>
#endif

#include "rendering/threads/WorkerThreadController.h"
#include "rendering/threads/work/SystemWorker.h"
#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "rendering/vulkan/helpers/VulkanRenderPass.h"
#include <simulation/runners/sim_vulkan_ticked_runner/ISimVulkanTickedRunner.h>

template<typename DeviceSelectorType>
vk::PhysicalDevice selectDevice(const vk::UniqueInstance& instance, DeviceSelectorType selector) {
    auto devices = instance->enumeratePhysicalDevices();

    for (const auto& device : devices) {
        if (selector(device))
            return device;
    }
    FATAL_ERROR("Could not find a suitable device.\n");
}

VulkanSimApp::VulkanSimApp(const vk::ApplicationInfo& appInfo, Size<uint32_t> windowSize)
    : context(appInfo, windowSize),
      device(*context.device),
      imguiRenderPass(device, context.surfaceFormat.format, VulkanRenderPass::Position::PipelineStartAndEnd, vk::ImageLayout::ePresentSrcKHR),
      simRenderPass(device, vk::Format::eR8G8B8A8Srgb, VulkanRenderPass::Position::PipelineStartAndEnd, vk::ImageLayout::eShaderReadOnlyOptimal),
      swapchain(context, imguiRenderPass)
{

    {
        // TODO - Make init order consistent with declare order
        //  objects are declared in VulkanWindow.h in a specific order so parents are destroyed after children
        //  i.e. command buffers are destroyed *before* the pool is destroyed
        //  so the pool is declared before it's children.
        auto poolInfo = vk::CommandPoolCreateInfo();
        poolInfo.queueFamilyIndex = context.queueFamilies.graphicsFamily;
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer; // Allow command buffers to be reset outside of the pool?

        cmdPool = device.createCommandPoolUnique(poolInfo);

        perFrameCommandBuffers.clear();
        auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
        cmdBufferAlloc.commandPool = *cmdPool;
        cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
        cmdBufferAlloc.commandBufferCount = swapchain.imageCount;
        perFrameCommandBuffers = device.allocateCommandBuffersUnique(cmdBufferAlloc);
    }
    
    {
        std::vector<vk::DescriptorPoolSize> pool_sizes =
                {
                        { vk::DescriptorType::eSampler, 1000 },
                        { vk::DescriptorType::eCombinedImageSampler, 1000 },
                        { vk::DescriptorType::eSampledImage, 1000 },
                        { vk::DescriptorType::eStorageImage, 1000 },
                        { vk::DescriptorType::eUniformTexelBuffer, 1000 },
                        { vk::DescriptorType::eStorageTexelBuffer, 1000 },
                        { vk::DescriptorType::eUniformBuffer, 1000 },
                        { vk::DescriptorType::eStorageBuffer, 1000 },
                        { vk::DescriptorType::eUniformBufferDynamic, 1000 },
                        { vk::DescriptorType::eStorageBufferDynamic, 1000 },
                        { vk::DescriptorType::eInputAttachment, 1000 }
                };
        vk::DescriptorPoolCreateInfo pool_info = {};
        pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;// VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * pool_sizes.size();
        pool_info.poolSizeCount = pool_sizes.size();
        pool_info.pPoolSizes = pool_sizes.data();
        descriptorPool = device.createDescriptorPoolUnique(pool_info);
    }

    {
        semaphores = std::make_unique<VulkanSimSemaphoreSet>(device);
        graphicsFence = std::make_unique<VulkanFence>(device);
    }
}

void VulkanSimApp::main_loop(SimulationBackendEnum backendType, const FluidParams &params, const SimSnapshot &snapshot) {
    pipelines = std::make_unique<VulkanSimPipelineSet>(device, *simRenderPass, Size<uint32_t>{snapshot.simSize.padded_pixel_size.x, snapshot.simSize.padded_pixel_size.y});

    auto systemWorker = SystemWorkerThreadController(std::make_unique<SystemWorkerThread>(*this, snapshot.simSize));
    auto simulationRunner = ISimVulkanTickedRunner::getForBackend(
            backendType,
            device, context.physicalDevice, *semaphores->renderFinishedShouldSim, *semaphores->simFinished
    );
    auto vulkanBuffers = simulationRunner->prepareBackend(params, snapshot);
    pipelines->buildSimulationFragDescriptors(device, *descriptorPool, vulkanBuffers);

    std::array<float, 32> frameTimes{};
    uint32_t currentFrame = 0;

    bool wantsRunSim = false;

    bool wantsQuit = false;
    while (!wantsQuit) {
        auto frameStartTime = std::chrono::steady_clock::now();
        //fprintf(stderr, "\nFrame start\n");
        uint32_t swFrameIndex;
        //fprintf(stderr, "Sending acquireNextImage (Signalling hasImage)\n");

        // NOTE - Previously the sim waited on imageCanBeChanged. This caused a complete stall.
        // I believe this occurs because imageCanBeChanged is a semaphore for "the display manager has stopped using *the current* image",
        // and that can only happen once another image has been given in.
        // See https://stackoverflow.com/a/52673669
        device.acquireNextImageKHR(*swapchain, std::numeric_limits<uint64_t>::max(), *semaphores->imageCanBeChanged, nullptr, &swFrameIndex);

        //fprintf(stderr, "Sending SystemWorker work\n");
        systemWorker.giveNextWork(SystemWorkerIn{
                .swFrameIndex = swFrameIndex,
                .swFramebuffer = *swapchain.framebuffers[swFrameIndex],
                .perf = {
                        .frameTimes = frameTimes,
                        .currentFrame = currentFrame
                }
        });

        // This dispatches the simulation, and signals the simFinished semaphore once it's done.
        // The simulation doesn't start until renderFinishedShouldSim is signalled, unless this is the first frame, at which point it doesn't bother waiting.
        simulationRunner->tick(1/60.0f, (currentFrame > 0), wantsRunSim);

        //fprintf(stderr, "Waiting on SystemWorker work\n");
        SystemWorkerOut systemOutput = systemWorker.getOutput();
        if (systemOutput.wantsQuit)
            wantsQuit = true;
        wantsRunSim = systemOutput.wantsRunSim;

        vk::SubmitInfo submitInfo{};
        vk::Semaphore waitSemaphores[] = {*semaphores->imageCanBeChanged, *semaphores->simFinished};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTopOfPipe};
        submitInfo.waitSemaphoreCount = 2;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        vk::CommandBuffer cmdBuffers[] = {
            systemOutput.cmdBuffer//*perFrameCommandBuffers[swFrameIndex]
        };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = cmdBuffers;

        vk::Semaphore signalSemaphores[] = {*semaphores->renderFinishedShouldPresent, *semaphores->renderFinishedShouldSim};
        submitInfo.signalSemaphoreCount = 2;
        submitInfo.pSignalSemaphores = signalSemaphores;

        //fprintf(stderr, "Submitting graphics work (Waiting on simFinished, signalling renderFinished)\n");
        if (currentFrame > 0) {
            // Now wait on the fence, to make sure we don't try to render two graphics at once
            device.waitForFences({**graphicsFence}, true, UINT64_MAX);
            // Reset the fence so we can use it again later
            device.resetFences({**graphicsFence});
        }
        context.graphicsQueue.submit({submitInfo}, **graphicsFence);


        vk::PresentInfoKHR presentInfo{};
        vk::Semaphore presentWaitSemaphores[] = {*semaphores->renderFinishedShouldPresent};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = presentWaitSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &(*swapchain);
        presentInfo.pImageIndices = &swFrameIndex;
        presentInfo.pResults = nullptr;
        //fprintf(stderr, "Submitting presentation (Waiting on renderFinished)\n");
        context.presentQueue.presentKHR(presentInfo);

        auto frameEndTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> frameTimeDiff = frameEndTime - frameStartTime;
        //printf("Frame %zu Time: %f\n", currentFrame % frameTimes.size(), frameTimeDiff.count());
        frameTimes[currentFrame % frameTimes.size()] = (frameTimeDiff).count();
        currentFrame++;
    }

    device.waitIdle();
}

#if CUDA_ENABLED
#include "simulation/memory/vulkan/VulkanSimulationAllocator.h"

SimSnapshot VulkanSimApp::test_cuda_sim(const FluidParams &params, const SimSnapshot &snapshot) {
    VulkanSimulationAllocator<CudaVulkan2DAllocator> allocator(device, context.physicalDevice);
    auto vulkanAllocs = allocator.makeAllocs(snapshot);

    auto sim = CudaBackendV1<false>(vulkanAllocs.simAllocs, params, snapshot);
    const float timeToRun = 10;
    float currentTime = 0;
    while(currentTime < timeToRun) {
        float maxTimestep = sim.findMaxTimestep();
        if (currentTime + maxTimestep > timeToRun)
            maxTimestep = timeToRun - currentTime;
        fprintf(stdout, "t: %f\tts: %f\r", currentTime, maxTimestep);
        sim.tick(maxTimestep);
        currentTime += maxTimestep;
    }
    fprintf(stdout, "\n");
    return sim.get_snapshot();
}
#endif