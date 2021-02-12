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


VulkanSimApp::VulkanSimApp(const vk::ApplicationInfo& appInfo, Size<uint32_t> windowSize)
    : context(appInfo, windowSize),
      device(*context.device),
      imguiRenderPass(device, context.surfaceFormat.format, VulkanRenderPass::Position::PipelineStartAndEnd, vk::ImageLayout::ePresentSrcKHR),
      simRenderPass(device, vk::Format::eR8G8B8A8Srgb, VulkanRenderPass::Position::PipelineStartAndEnd, vk::ImageLayout::eShaderReadOnlyOptimal),
      swapchain(context, imguiRenderPass)
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
        ImGui_ImplVulkan_Init(&init_info, *imguiRenderPass);

        // Allocate a command buffer to create the font texture for ImGui
        {
            auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
            cmdBufferAlloc.commandPool = *context.cmdPool;
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
        device, *simRenderPass, snapshot.simSize.padded_pixel_size
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

        .imguiRenderPass = *imguiRenderPass,
        .imguiRenderArea = vk::Rect2D({0,0}, {context.windowSize.x, context.windowSize.y}),
        .simRenderPass = *simRenderPass,
        .simRenderArea = vk::Rect2D({0,0}, {snapshot.simSize.padded_pixel_size.x, snapshot.simSize.padded_pixel_size.y}),

        .pipelines=pipelines
    }, vulkanAllocator->vulkanFrames, swapchain);
//    fprintf(stderr, "created vulkansimappdata\n");

    simulationRunner->prepareSemaphores(data);
//    fprintf(stderr, "created semaphores\n");

    auto systemWorker = SystemWorkerThreadController(std::make_unique<SystemWorkerThread>(data));
//    fprintf(stderr, "created systemworker\n");

    uint32_t renderedFrameNum = 0;
    uint32_t simFrameIdx = 0;
    uint32_t swapchainImageIdx = 0;

    std::array<float, 32> frameTimes{};
    bool wantsRunSim = false;
    bool wantsQuit = false;

//    fprintf(stderr, "starting loop\n");
    // Store the time the current frame started.
    // When a new frame starts, the delta time will be counted as that frame's length.
    // We will always submit frames and present them at an approximately equal rate, so this is an accurate FPS measure.
    auto lastFrameStartTime = std::chrono::steady_clock::now();
    while (!wantsQuit) {
        // Record how long the last frame took
        auto frameStartTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> frameTimeDiff = frameStartTime - lastFrameStartTime;
        frameTimes[renderedFrameNum % frameTimes.size()] = (frameTimeDiff).count();
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
                        .currentFrameNum = renderedFrameNum
                }
        });
//        fprintf(stderr, "gave systemworker work\n");

        // This dispatches the simulation, and signals the simFinished semaphore once it's done.
        // The simulation doesn't start until renderFinishedShouldSim is signalled, unless this is the first frame, at which point it doesn't bother waiting.
        simulationRunner->tick(
            1/60.0f, // Simulate 1/60th of a second
            !simFrameIsFresh, // Only wait for the render if this sim frame has started rendering before
            wantsRunSim, // Actually run the sim or not
            simFrameIdx
        );
//        fprintf(stderr, "ticked sim\n");

        // Simulation is done, grab the thread output
        SystemWorkerOut systemOutput = systemWorker.getOutput();
        if (systemOutput.wantsQuit)
            wantsQuit = true;
        wantsRunSim = systemOutput.wantsRunSim;
//        fprintf(stderr, "got output\n");

        // Send the graphics work
        {
            vk::SubmitInfo submitInfo{};
            vk::Semaphore waitSemaphores[] = {*simFrame.imageAcquired, *simFrame.simFinished};
            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTopOfPipe};
            submitInfo.waitSemaphoreCount = 2;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            vk::CommandBuffer cmdBuffers[] = {
                systemOutput.cmdBuffer
            };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = cmdBuffers;

            vk::Semaphore signalSemaphores[] = {*simFrame.renderFinishedShouldPresent, *simFrame.renderFinishedShouldSim};
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