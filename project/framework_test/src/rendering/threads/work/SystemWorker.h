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

    VulkanBackedFramebuffer simFramebuffer;
    vk::UniqueDescriptorSet simImageDescriptorSet;

    ImGuiContext* context;

    // TODO - make this allocate the command pool itself?
public:
    explicit SystemWorker(VulkanSimApp& vulkanWindow, SimSize simSize)
        : window(vulkanWindow.context.window),
          imguiRenderPass(*vulkanWindow.imguiRenderPass),
          simRenderPass(*vulkanWindow.simRenderPass),
          imguiRenderArea({0, 0}, {vulkanWindow.context.windowSize.x, vulkanWindow.context.windowSize.y}),
          simRenderArea({0, 0}, {simSize.padded_pixel_size.x, simSize.padded_pixel_size.y}),
          pipelines(vulkanWindow.pipelines.get()),
          simSize(simSize),
          simFragPushConstants({
                  .pixelWidth=simSize.padded_pixel_size.x,
                  .pixelHeight=simSize.padded_pixel_size.y,
                  .columnStride=simSize.padded_pixel_size.y, // TODO
                  .totalPixels=(uint32_t)simSize.pixel_count()
          }),
          simFramebuffer(vulkanWindow.context,
                         vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                         simSize.padded_pixel_size,
                         *vulkanWindow.simRenderPass
                         ),
           context(vulkanWindow.imContext)
    {
        // Allocate command buffers
        // TODO don't do this here
        {
            auto cmdBufferAlloc = vk::CommandBufferAllocateInfo();
            cmdBufferAlloc.commandPool = *vulkanWindow.cmdPool;
            cmdBufferAlloc.level = vk::CommandBufferLevel::ePrimary;
            cmdBufferAlloc.commandBufferCount = vulkanWindow.swapchain.imageCount;
            frameCmdBuffers = vulkanWindow.device.allocateCommandBuffersUnique(cmdBufferAlloc);
        }

        vk::DescriptorSet descriptorSet = ImGui_ImplVulkan_MakeDescriptorSet(simFramebuffer.getImageView());
        simImageDescriptorSet = vk::UniqueDescriptorSet(descriptorSet, vk::PoolFree(vulkanWindow.device, *vulkanWindow.descriptorPool, VULKAN_HPP_DEFAULT_DISPATCHER));
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
        ImGui::Image((ImTextureID)*simImageDescriptorSet, ImVec2(simSize.padded_pixel_size.x, simSize.padded_pixel_size.y+2));
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