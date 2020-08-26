//
// Created by samuel on 22/08/2020.
//

#pragma once

#include "BaseThread.h"

#include <SDL_vulkan.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include "rendering/vulkan/VulkanWindow.h"


struct SystemWorkerIn {
    uint32_t swFrameIndex;
    vk::Framebuffer targetFramebuffer;
};

struct SystemWorkerOut {
    bool wantsQuit = false;
    vk::CommandBuffer cmdBuffer;
};

/**
 * Processes SDL input, and builds the command buffer for the next frame
 */
class SystemThreadWorker : public IThreadWorker<SystemWorkerIn, SystemWorkerOut> {
    SDL_Window* window;
    vk::RenderPass renderPass;
    vk::Rect2D renderArea;
    VulkanPipelineSet* pipelines;
    std::vector<vk::UniqueCommandBuffer> frameCmdBuffers;

    // TODO - make this allocate the command pool itself?
public:
    explicit SystemThreadWorker(const VulkanWindow& vulkanWindow)
        : window(vulkanWindow.window),
          renderPass(*vulkanWindow.renderPass),
          renderArea({0, 0}, {(uint32_t)vulkanWindow.window_size.x, (uint32_t)vulkanWindow.window_size.y}),
          pipelines(vulkanWindow.pipelines.get())
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplSDL2_InitForVulkan(window);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = *vulkanWindow.instance;
        init_info.PhysicalDevice = vulkanWindow.physicalDevice;
        init_info.Device = *vulkanWindow.logicalDevice;
        init_info.QueueFamily = vulkanWindow.queueFamilies.graphics_family.value();
        init_info.Queue = vulkanWindow.graphicsQueue;
        init_info.PipelineCache = nullptr;
        init_info.DescriptorPool = *vulkanWindow.descriptorPool;
        init_info.Allocator = nullptr;
        init_info.MinImageCount = vulkanWindow.swapchainProps.imageCount; // TODO - this isn't right
        init_info.ImageCount = vulkanWindow.swapchainProps.imageCount;
        init_info.CheckVkResultFn = nullptr; // TODO
        ImGui_ImplVulkan_Init(&init_info, renderPass);

        {
            auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
            cmdBufferAlloc.commandPool = *vulkanWindow.cmdPool;
            cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
            cmdBufferAlloc.commandBufferCount = 1 + vulkanWindow.swapchainProps.imageCount;
            frameCmdBuffers = vulkanWindow.logicalDevice->allocateCommandBuffersUnique(cmdBufferAlloc);
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
            vulkanWindow.graphicsQueue.submit({submitInfo}, nullptr);
            vulkanWindow.graphicsQueue.waitIdle();
            ImGui_ImplVulkan_DestroyFontUploadObjects();
        }
    }
    ~SystemThreadWorker() override {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
    }

    void threadLoop() override {
        bool showDemoWindow = true;

        while(true) {
            SystemWorkerIn input = waitForInput();

            bool wantsQuit = false;
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    wantsQuit = true;
                else
                    ImGui_ImplSDL2_ProcessEvent(&event);
            }

            const auto& cmdBuffer = *frameCmdBuffers[input.swFrameIndex];
            cmdBuffer.reset({});

            auto beginInfo = vk::CommandBufferBeginInfo();
            auto renderPassInfo = vk::RenderPassBeginInfo();
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = input.targetFramebuffer;
            renderPassInfo.renderArea = renderArea;
            auto clearColor = vk::ClearValue(vk::ClearColorValue());
            clearColor.color.setFloat32({1.0f, 0.0f, 1.0f, 1.0f});
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            cmdBuffer.begin(beginInfo);
            cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines->redTriangle);
            cmdBuffer.draw(3, 1, 0, 0);

            {
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplSDL2_NewFrame(window);
                ImGui::NewFrame();

                if (showDemoWindow)
                    ImGui::ShowDemoWindow(&showDemoWindow);

                ImGui::Render();

                ImDrawData* draw_data = ImGui::GetDrawData();
                ImGui_ImplVulkan_RenderDrawData(draw_data, cmdBuffer);
            }

            cmdBuffer.endRenderPass();
            cmdBuffer.end();

            pushOutput(SystemWorkerOut{
                    .wantsQuit = wantsQuit,
                    .cmdBuffer = cmdBuffer
            });
        }
    }
};