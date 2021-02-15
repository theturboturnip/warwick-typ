//
// Created by samuel on 22/08/2020.
//
#include "VulkanSimApp.h"

#include "rendering/vulkan/helpers/VulkanDebug.h"

#if CUDA_ENABLED
#include <simulation/backends/cuda/CudaBackendV1.cuh>
#endif

#include "rendering/threads/WorkerThreadController.h"
#include "rendering/threads/work/SystemWorker.h"
#include "rendering/vulkan/helpers/VulkanQueueFamilies.h"
#include "rendering/vulkan/helpers/VulkanRenderPass.h"
#include <simulation/runners/sim_vulkan_ticked_runner/ISimVulkanTickedRunner.h>

#include "memory/FrameSetAllocator.h"


VulkanSimApp::VulkanSimApp(const vk::ApplicationInfo& appInfo, SimAppProperties props, Size<uint32_t> windowSize)
    : props(props),
      context(appInfo, windowSize, props.useVsync),
      device(*context.device),
      finalCompositeRenderPass(device, context.surfaceFormat.format, VulkanRenderPass::Position::PipelineStartAndEnd, vk::ImageLayout::ePresentSrcKHR),
      vizRenderPass(device, vk::Format::eR8G8B8A8Unorm, VulkanRenderPass::Position::PipelineStartAndEnd, vk::ImageLayout::eShaderReadOnlyOptimal),
      swapchain(context, finalCompositeRenderPass)
{
    {
        // Init ImGUI
        IMGUI_CHECKVERSION();
        imContext = ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplSDL2_InitForVulkan(context.window);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = *context.instance;
        init_info.PhysicalDevice = context.physicalDevice;
        init_info.Device = *context.device;
        init_info.QueueFamily = context.queueFamilies.graphicsFamily;
        init_info.Queue = context.graphicsQueue;
        init_info.PipelineCache = nullptr;
        init_info.DescriptorPool = *context.descriptorPool;
        init_info.Allocator = nullptr;
        init_info.MinImageCount = swapchain.imageCount; // TODO - this isn't right
        init_info.ImageCount = swapchain.imageCount;
        init_info.CheckVkResultFn = nullptr; // TODO
        ImGui_ImplVulkan_Init(&init_info, *finalCompositeRenderPass);

        // Allocate a command buffer to create the font texture for ImGui
        {
            auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
            cmdBufferAlloc.commandPool = *context.graphicsCmdPool;
            cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
            cmdBufferAlloc.commandBufferCount = 1;
            const auto fontCmdBuffer = std::move(device.allocateCommandBuffersUnique(cmdBufferAlloc)[0]);

            vk::CommandBufferBeginInfo begin_info = {};
            begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
            {
                fontCmdBuffer->begin(begin_info);
                ImGui_ImplVulkan_CreateFontsTexture(*fontCmdBuffer);
                fontCmdBuffer->end();
            }

            vk::SubmitInfo submitInfo = {};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &(*fontCmdBuffer);
            context.graphicsQueue.submit({submitInfo}, nullptr);
            context.graphicsQueue.waitIdle();
            ImGui_ImplVulkan_DestroyFontUploadObjects();
        }
    }
}

VulkanSimApp::~VulkanSimApp() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext(imContext);
}

void VulkanSimApp::main_loop(SimulationBackendEnum backendType, const FluidParams &params, const SimSnapshot &snapshot) {
//    fprintf(stderr, "Starting main loop\n");
    // At max, we can render two frames at once - render n, while simulating n+1
    const size_t maxFramesInFlight = 2;

    auto pipelines = VulkanSimPipelineSet(
            device, *vizRenderPass, snapshot.simSize.padded_pixel_size
    );

//    fprintf(stderr, "Created pipelines\n");

    auto simulationRunner = ISimVulkanTickedRunner::getForBackend(
            backendType,
            context
    );
    VulkanFrameSetAllocator* vulkanAllocator = simulationRunner->prepareBackend(params, snapshot, maxFramesInFlight);
//    fprintf(stderr, "created backend and allocator\n");

    VulkanSimAppData data(VulkanSimAppData::Global{
        .context = context,
        .imguiContext = imContext,

        .simSize = snapshot.simSize,

        .finalCompositeRenderPass = *finalCompositeRenderPass,
        .finalCompositeRect = vk::Rect2D({0, 0}, {context.windowSize.x, context.windowSize.y}),
        .vizRenderPass = *vizRenderPass,
        .vizRect = vk::Rect2D({0, 0}, {snapshot.simSize.padded_pixel_size.x*2, snapshot.simSize.padded_pixel_size.y*2}),

        .pipelines=pipelines
    }, vulkanAllocator->vulkanFrames, swapchain);
//    fprintf(stderr, "created vulkansimappdata\n");

    simulationRunner->prepareSemaphores(data);
//    fprintf(stderr, "created semaphores\n");

    auto systemWorker = SystemWorkerThreadController(std::make_unique<SystemWorkerThread>(data));
//    fprintf(stderr, "created systemworker\n");

    // Number of total rendered/simulated frames
    uint32_t renderedFrameNum = 0;
    uint32_t simFrameNum = 0;

    // Index of the simulation frame buffer to use
    uint32_t simFrameIdx = 0;
    // Index of the swapchain image to use
    uint32_t swapchainImageIdx = 0;

    // User Input Values
    bool wantsRunSim = false;
    bool wantsQuit = false;

    // Time taken to render overall
    std::array<float, 32> frameTimes{};
    // Sim-time taken in previous sim ticks
    std::array<float, 32> simTickLengths{};
    // Real-time taken to simulate sim-ticks
    std::array<float, 32> simFrameTimes{};

    double elapsedSimTime = 0.0;
    double elapsedRealTime = 0.0;
    double elapsedRealTimeDuringSim = 0.0;

//    fprintf(stderr, "starting loop\n");
    // Store the time the current frame started.
    // When a new frame starts, the delta time will be counted as that frame's length.
    // We will always submit frames and present them at an approximately equal rate, so this is an accurate FPS measure.
    auto lastFrameStartTime = std::chrono::steady_clock::now();
    while (!wantsQuit) {
        // Record how long the last frame took
        auto frameStartTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> frameTimeDiff = frameStartTime - lastFrameStartTime;

        frameTimes[renderedFrameNum % frameTimes.size()] = frameTimeDiff.count();
        elapsedRealTime += frameTimeDiff.count();
        if (wantsRunSim) {
            elapsedRealTimeDuringSim += frameTimeDiff.count();
            simFrameTimes[simFrameNum % frameTimes.size()] = frameTimeDiff.count();
        }
//        fprintf(stderr, "added frame time\n");
        lastFrameStartTime = frameStartTime;

        // Get a new sim frame
        auto& simFrame = data.frameData[simFrameIdx];
        // If we haven't yet kicked off a render job on this frame, it's considered "fresh".
        // That means we shouldn't wait on semaphores etc.
        const bool simFrameIsFresh = (renderedFrameNum < maxFramesInFlight);

        // Get a new swapchain image
        device.acquireNextImageKHR(
            *swapchain,
            UINT64_MAX,
            *simFrame.imageAcquired, nullptr,
            &swapchainImageIdx
        );
//        fprintf(stderr, "told to get swapchain image %d\n", swapchainImageIdx);
        auto& swapchainImage = data.swapchainImageData[swapchainImageIdx];
        // Wait for the last "sim frame" that was rendering this swapchain image to finish
        if (swapchainImage.inFlight) {
            device.waitForFences({swapchainImage.inFlight}, true, UINT64_MAX);
        }
        // We are now trying to render this image, so make the swapchain fence point to the simframe fence.
        swapchainImage.inFlight = *simFrame.inFlight;
//        fprintf(stderr, "got sim frame and swapchain image\n");

        // Enqueue work for the SystemWorker
        systemWorker.giveNextWork(SystemWorkerIn{
                .swapchainImageIndex = swapchainImageIdx,
                .simFrameIndex = simFrameIdx,
                .perf = {
                        .frameTimes = frameTimes,
                        .currentFrameNum = renderedFrameNum,

                        .simFrameTimes = simFrameTimes,
                        .simTickLengths = simTickLengths,
                        .simFrameNum = simFrameNum,

                        .elapsedRealTime = elapsedRealTime,
                        .elapsedSimTime = elapsedSimTime,
                        .elapsedRealTimeDuringSim = elapsedRealTimeDuringSim
                }
        });
//        fprintf(stderr, "gave systemworker work\n");

        // TODO Decide on the next simulation tick length.
        const float simTickLength = 1/60.0f;

        // This dispatches the simulation, and signals the simFinished semaphore once it's done.
        // The simulation doesn't start until renderFinishedShouldSim is signalled, unless this is the first frame, at which point it doesn't bother waiting.
        simulationRunner->tick(
            simTickLength,
            !simFrameIsFresh, // Only wait for the render if this sim frame has started rendering before
            wantsRunSim, // Actually run the sim or not
            simFrameIdx
        );
//        fprintf(stderr, "ticked sim\n");

        if (wantsRunSim) {
            elapsedSimTime += simTickLength;
            simTickLengths[simFrameNum % simTickLengths.size()] = simTickLength;
        }

        // Simulation is done, grab the thread output
        SystemWorkerOut systemOutput = systemWorker.getOutput();
        if (systemOutput.wantsQuit)
            wantsQuit = true;
        wantsRunSim = systemOutput.wantsRunSim;
//        fprintf(stderr, "got output\n");

        // Send the compute work
        {
            // We need to make sure the compute waits for the render to finish.
            //  example
            //    | sim1 | compute1 | sim2 | compute2 |  <- overlaps with render, race condition
            //                      |   render1   |
            //  with the semaphore:
            //    | sim1 | compute1 | sim2 |      | compute2 |  <- no overlap
            //                      |   render1   |
            vk::SubmitInfo submitInfo{};
            std::vector<vk::Semaphore> waitSemaphores = {*simFrame.simFinished};
            std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eComputeShader};
            if (!simFrameIsFresh) {
                waitSemaphores.push_back(*simFrame.renderFinishedShouldCompute);
                waitStages.push_back(vk::PipelineStageFlagBits::eComputeShader);
            }
            submitInfo.waitSemaphoreCount = waitSemaphores.size();
            submitInfo.pWaitSemaphores = waitSemaphores.data();
            submitInfo.pWaitDstStageMask = waitStages.data();

            vk::CommandBuffer cmdBuffers[] = {
                    systemOutput.computeCmdBuffer
            };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = cmdBuffers;

            vk::Semaphore signalSemaphores[] = {*simFrame.computeFinished, *simFrame.computeFinishedShouldSim};
            submitInfo.signalSemaphoreCount = 2;
            submitInfo.pSignalSemaphores = signalSemaphores;

            context.computeQueue.submit({submitInfo}, nullptr);
        }

        // Send the graphics work
        {
            vk::SubmitInfo submitInfo{};
            vk::Semaphore waitSemaphores[] = {*simFrame.imageAcquired, *simFrame.computeFinished};
            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTopOfPipe};
            submitInfo.waitSemaphoreCount = 2;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            vk::CommandBuffer cmdBuffers[] = {
                systemOutput.graphicsCmdBuffer
            };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = cmdBuffers;

            vk::Semaphore signalSemaphores[] = {*simFrame.renderFinishedShouldPresent, *simFrame.renderFinishedShouldCompute};
            submitInfo.signalSemaphoreCount = 2;
            submitInfo.pSignalSemaphores = signalSemaphores;

            // Reset this frame's fence before we submit
            device.resetFences({*simFrame.inFlight});
            // and tell the graphics queue to re-open the fence once it's done with this submission
            context.graphicsQueue.submit({submitInfo}, *simFrame.inFlight);
        }

        // Send the Present work
        {
            vk::PresentInfoKHR presentInfo{};
            vk::Semaphore presentWaitSemaphores[] = {*simFrame.renderFinishedShouldPresent};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = presentWaitSemaphores;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &(*swapchain);
            presentInfo.pImageIndices = &swapchainImageIdx;
            presentInfo.pResults = nullptr;
            context.presentQueue.presentKHR(presentInfo);
        }

        // Increment renderedFrameNum without bounding
        renderedFrameNum++;
        if (wantsRunSim) {
            // Increment simFrameNum without bounding
            simFrameNum++;
        }
        // Increment simFrameIdx with looping
        simFrameIdx = (simFrameIdx + 1) % maxFramesInFlight;
    }

    device.waitIdle();
}

#if CUDA_ENABLED
SimSnapshot VulkanSimApp::test_cuda_sim(const FluidParams &params, const SimSnapshot &snapshot) {
    const size_t frameCount = 1;
    auto allocator = FrameSetAllocator<MType::VulkanCuda, CudaBackendV1<false>::Frame>(
            context, snapshot.simSize.padded_pixel_size, frameCount
    );

    int frameToWriteIdx = 0;

    auto sim = CudaBackendV1<false>(allocator.frames, params, snapshot);
    const float timeToRun = 10;
    float currentTime = 0;
    while(currentTime < timeToRun) {
        float maxTimestep = sim.findMaxTimestep();
        if (currentTime + maxTimestep > timeToRun)
            maxTimestep = timeToRun - currentTime;
        fprintf(stdout, "t: %f\tts: %f\r", currentTime, maxTimestep);
        sim.tick(maxTimestep, frameToWriteIdx);
        currentTime += maxTimestep;
        frameToWriteIdx = (frameToWriteIdx + 1) % frameCount;
    }
    fprintf(stdout, "\n");
    return sim.get_snapshot();
}
#endif