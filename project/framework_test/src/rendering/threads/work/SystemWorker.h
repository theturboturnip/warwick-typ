//
// Created by samuel on 22/08/2020.
//

#pragma once

#include "rendering/vulkan/VulkanSimApp.h"
#include <SDL_vulkan.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <rendering/vulkan/helpers/VulkanDeviceMemory.h>


struct SystemWorkerIn {
    uint32_t swFrameIndex;
    vk::Framebuffer swFramebuffer;

    struct PerfData {
        std::array<float, 32> frameTimes;
        uint32_t currentFrame;
    } perf;
};

struct SystemWorkerOut {
    bool wantsQuit = false;
    bool wantsRunSim;
    vk::CommandBuffer cmdBuffer;
};

/**
 * Processes SDL input, and builds the command buffer for the next frame
 */
class SystemWorker {
    SDL_Window* window;
    vk::RenderPass imguiRenderPass;
    vk::RenderPass simRenderPass;
    vk::Rect2D imguiRenderArea;
    vk::Rect2D simRenderArea;
    VulkanSimPipelineSet * pipelines;
    std::vector<vk::UniqueCommandBuffer> frameCmdBuffers;
    bool showDemoWindow = true;
    bool wantsRunSim = false;

    SimSize simSize;

    VulkanSimPipelineSet::SimFragPushConstants simFragPushConstants;

    vk::UniqueImage simFramebufferImage;
    VulkanDeviceMemory simFramebufferMemory;
    vk::UniqueImageView simFramebufferImageView;
    vk::UniqueFramebuffer simFramebuffer;
    vk::UniqueDescriptorSet simImageDescriptorSet;

    ImGuiContext* context;

    // TODO - make this allocate the command pool itself?
public:
    explicit SystemWorker(const VulkanSimApp & vulkanWindow, SimSize simSize)
        : window(vulkanWindow.setup.window),
          imguiRenderPass(*vulkanWindow.imguiRenderPass),
          simRenderPass(*vulkanWindow.simRenderPass),
          imguiRenderArea({0, 0}, {vulkanWindow.setup.windowSize.x, vulkanWindow.setup.windowSize.y}),
          simRenderArea({0, 0}, {simSize.pixel_size.x+2, simSize.pixel_size.y+2}),
          pipelines(vulkanWindow.pipelines.get()),
          simSize(simSize),
          simFragPushConstants({
                  .pixelWidth=simSize.pixel_size.x+2,
                  .pixelHeight=simSize.pixel_size.y+2,
                  .columnStride=simSize.pixel_size.y+2, // TODO
                  .totalPixels=(uint32_t)simSize.pixel_count()
          })
    {
        IMGUI_CHECKVERSION();
        context = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplSDL2_InitForVulkan(window);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = *vulkanWindow.setup.instance;
        init_info.PhysicalDevice = vulkanWindow.setup.physicalDevice;
        init_info.Device = *vulkanWindow.setup.device;
        init_info.QueueFamily = vulkanWindow.setup.queueFamilies.graphicsFamily;
        init_info.Queue = vulkanWindow.setup.graphicsQueue;
        init_info.PipelineCache = nullptr;
        init_info.DescriptorPool = *vulkanWindow.descriptorPool;
        init_info.Allocator = nullptr;
        init_info.MinImageCount = vulkanWindow.swapchain.imageCount; // TODO - this isn't right
        init_info.ImageCount = vulkanWindow.swapchain.imageCount;
        init_info.CheckVkResultFn = nullptr; // TODO
        ImGui_ImplVulkan_Init(&init_info, imguiRenderPass);

        {
            auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
            cmdBufferAlloc.commandPool = *vulkanWindow.cmdPool;
            cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
            cmdBufferAlloc.commandBufferCount = 1 + vulkanWindow.swapchain.imageCount;
            frameCmdBuffers = vulkanWindow.device.allocateCommandBuffersUnique(cmdBufferAlloc);
            const auto fontCmdBuffer = std::move(frameCmdBuffers.back());
            frameCmdBuffers.pop_back();

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
            vulkanWindow.setup.graphicsQueue.submit({submitInfo}, nullptr);
            vulkanWindow.setup.graphicsQueue.waitIdle();
            ImGui_ImplVulkan_DestroyFontUploadObjects();
        }

        // TODO - make simFramebuffer elements
        {
            auto imageCreateInfo = vk::ImageCreateInfo{};
            imageCreateInfo.imageType = vk::ImageType::e2D;
            imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
            imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
            imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
            imageCreateInfo.extent.width = simSize.pixel_size.x+2;
            imageCreateInfo.extent.height = simSize.pixel_size.y+2;
            imageCreateInfo.extent.depth = 1;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.format = vk::Format::eR8G8B8A8Srgb;
            imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
            imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
            simFramebufferImage = vulkanWindow.device.createImageUnique(imageCreateInfo);

            vk::MemoryRequirements memRequirements = vulkanWindow.device.getImageMemoryRequirements(*simFramebufferImage);

            simFramebufferMemory = VulkanDeviceMemory(
                    vulkanWindow.device,
                    vulkanWindow.setup.physicalDevice,
                    memRequirements,
                    vk::MemoryPropertyFlagBits::eDeviceLocal
                    );

            vulkanWindow.device.bindImageMemory(*simFramebufferImage, *simFramebufferMemory, 0);

            // TODO - refactor into common function with VulkanSwapchain
            auto makeIdentityView = [&](vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspectFlags = vk::ImageAspectFlagBits::eColor){
                auto createInfo = vk::ImageViewCreateInfo();
                createInfo.image = image;
                createInfo.viewType = vk::ImageViewType::e2D;
                createInfo.format = format;

                createInfo.components.r = vk::ComponentSwizzle::eIdentity;
                createInfo.components.g = vk::ComponentSwizzle::eIdentity;
                createInfo.components.b = vk::ComponentSwizzle::eIdentity;
                createInfo.components.a = vk::ComponentSwizzle::eIdentity;

                createInfo.subresourceRange.aspectMask = aspectFlags;
                // We don't do any mipmapping/texture arrays ever - only use the first mip level, and the first array layer
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                return vulkanWindow.device.createImageViewUnique(createInfo);
            };
            simFramebufferImageView = makeIdentityView(*simFramebufferImage, imageCreateInfo.format);

            auto framebufferCreateInfo = vk::FramebufferCreateInfo();
            framebufferCreateInfo.renderPass = *vulkanWindow.simRenderPass;
            framebufferCreateInfo.attachmentCount = 1;
            framebufferCreateInfo.pAttachments = &(*simFramebufferImageView);
            framebufferCreateInfo.width = imageCreateInfo.extent.width;
            framebufferCreateInfo.height = imageCreateInfo.extent.height;
            framebufferCreateInfo.layers = 1;
            simFramebuffer = vulkanWindow.device.createFramebufferUnique(framebufferCreateInfo);

            vk::DescriptorSet descriptorSet = ImGui_ImplVulkan_MakeDescriptorSet(*simFramebufferImageView);
            simImageDescriptorSet = vk::UniqueDescriptorSet(descriptorSet, vk::PoolFree(vulkanWindow.device, *vulkanWindow.descriptorPool, VULKAN_HPP_DEFAULT_DISPATCHER));
        }
    }
    ~SystemWorker() {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext(context);
    }

    SystemWorkerOut work(SystemWorkerIn input) {
        bool wantsQuit = false;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                wantsQuit = true;
            else
                ImGui_ImplSDL2_ProcessEvent(&event);
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

//        if (showDemoWindow)
//            ImGui::ShowDemoWindow(&showDemoWindow);

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowBgAlpha(0.1f);
        // ImGuiWindowFlags_NoBackground
        ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
        {
            ImGui::Text("Frame %u", input.perf.currentFrame);
            if (input.perf.currentFrame >= input.perf.frameTimes.size()) {
                float sumFrameTimes = std::accumulate(input.perf.frameTimes.begin(), input.perf.frameTimes.end(), 0.0f);
                float avgFrameTime = sumFrameTimes / input.perf.frameTimes.size();
                ImGui::Text("Avg FPS: %.1f", 1.0f / avgFrameTime);
                ImGui::Text("Avg frame time: %.2fms", avgFrameTime*1000.0f);
            } else {
                ImGui::Text("Not enough frames for FPS");
            }
            ImGui::Checkbox("Running", &wantsRunSim);
        }
        ImGui::End();

        //ImGui::SetNextWindowSize(ImVec2(simSize.pixel_size.x+2, simSize.pixel_size.y+2));
        ImGui::Begin("Simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Image((ImTextureID)*simImageDescriptorSet, ImVec2(simSize.pixel_size.x+2, simSize.pixel_size.y+2));
        ImGui::End();
        ImGui::Render();

        const auto& cmdBuffer = *frameCmdBuffers[input.swFrameIndex];
        cmdBuffer.reset({});

        auto clearColor = vk::ClearValue(vk::ClearColorValue());
        clearColor.color.setFloat32({0.0f, 0.0f, 0.1f, 1.0f});


        auto beginInfo = vk::CommandBufferBeginInfo();

        cmdBuffer.begin(beginInfo);

        {
            auto simRenderPassInfo = vk::RenderPassBeginInfo();
            simRenderPassInfo.renderPass = simRenderPass;
            simRenderPassInfo.framebuffer = *simFramebuffer;
            simRenderPassInfo.renderArea = simRenderArea;
            simRenderPassInfo.clearValueCount = 1;
            simRenderPassInfo.pClearValues = &clearColor;
            cmdBuffer.beginRenderPass(simRenderPassInfo, vk::SubpassContents::eInline);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines->fullscreenPressure);
            cmdBuffer.pushConstants(
                    *pipelines->fullscreenPressure.layout,
                    vk::ShaderStageFlagBits::eFragment,
                    0,
                    vk::ArrayProxy<const VulkanSimPipelineSet::SimFragPushConstants>{simFragPushConstants});
            cmdBuffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics,
                    *pipelines->fullscreenPressure.layout,
                    0,
                    {*pipelines->simulationFragDescriptors},
                    {});
            cmdBuffer.draw(6, 1, 0, 0);
            cmdBuffer.endRenderPass();
        }

        {
            auto imguiRenderPassInfo = vk::RenderPassBeginInfo();
            imguiRenderPassInfo.renderPass = imguiRenderPass;
            imguiRenderPassInfo.framebuffer = input.swFramebuffer;
            imguiRenderPassInfo.renderArea = imguiRenderArea;
            imguiRenderPassInfo.clearValueCount = 1;
            imguiRenderPassInfo.pClearValues = &clearColor;
            cmdBuffer.beginRenderPass(imguiRenderPassInfo, vk::SubpassContents::eInline);


            {
                ImDrawData *draw_data = ImGui::GetDrawData();
                ImGui_ImplVulkan_RenderDrawData(draw_data, context, cmdBuffer, imguiRenderPass, VK_SAMPLE_COUNT_1_BIT);
            }

            cmdBuffer.endRenderPass();
        }
        cmdBuffer.end();

        return SystemWorkerOut{
            .wantsQuit = wantsQuit,
            .wantsRunSim = wantsRunSim,
            .cmdBuffer = cmdBuffer
        };
    }
};

#include "rendering/threads/IWorkerThread_Impl.inl"
using SystemWorkerThread = IWorkerThread_Impl<SystemWorker, SystemWorkerIn, SystemWorkerOut>;
#include "rendering/threads/WorkerThreadController.h"
using SystemWorkerThreadController = WorkerThreadController<SystemWorkerIn, SystemWorkerOut>;