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
    } perf;
};

struct SystemWorkerOut {
    bool wantsQuit = false;
    bool wantsRunSim = false;
    vk::CommandBuffer cmdBuffer;
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

public:
    explicit SystemWorker(VulkanSimAppData& data)
        : data(data),
          global(data.globalData),
          simBuffersPushConstants({
              .pixelWidth=global.simSize.padded_pixel_size.x,
              .pixelHeight=global.simSize.padded_pixel_size.y,
              .columnStride=global.simSize.padded_pixel_size.y, // TODO
              .totalPixels=(uint32_t)global.simSize.pixel_count()
          })
    {}

    SystemWorkerOut work(SystemWorkerIn input) {
        bool wantsQuit = false;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                wantsQuit = true;
            else
                ImGui_ImplSDL2_ProcessEvent(&event);
        }

        auto& simFrameData = data.frameData[input.simFrameIndex];
        auto& swImageData = data.swapchainImageData[input.swapchainImageIndex];

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame(global.context.window);
        ImGui::NewFrame();

//        if (showDemoWindow)
//            ImGui::ShowDemoWindow(&showDemoWindow);

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowBgAlpha(0.1f);
        // ImGuiWindowFlags_NoBackground
        ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
        {
            ImGui::Text("Frame %u", input.perf.currentFrameNum);
            if (input.perf.currentFrameNum >= input.perf.frameTimes.size()) {
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
        ImGui::Image(
                (ImTextureID)*simFrameData.vizFramebufferDescriptorSet,
                ImVec2(global.simSize.padded_pixel_size.x, global.simSize.padded_pixel_size.y)
        );
        ImGui::End();
        ImGui::Render();

        const auto& cmdBuffer = *simFrameData.threadOutputs.commandBuffer;
        cmdBuffer.reset({});

        auto clearColor = vk::ClearValue(vk::ClearColorValue());
        clearColor.color.setFloat32({0.0f, 0.0f, 0.1f, 1.0f});


        auto beginInfo = vk::CommandBufferBeginInfo();

        cmdBuffer.begin(beginInfo);

        {
            auto simRenderPassInfo = vk::RenderPassBeginInfo();
            simRenderPassInfo.renderPass = global.simRenderPass;
            simRenderPassInfo.framebuffer = *simFrameData.vizFramebuffer;
            simRenderPassInfo.renderArea = global.simRenderArea;
            simRenderPassInfo.clearValueCount = 1;
            simRenderPassInfo.pClearValues = &clearColor;
            cmdBuffer.beginRenderPass(simRenderPassInfo, vk::SubpassContents::eInline);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *global.pipelines.fullscreenPressure);
            cmdBuffer.pushConstants(
                    *global.pipelines.fullscreenPressure.layout,
                    vk::ShaderStageFlagBits::eFragment,
                    0,
                    vk::ArrayProxy<const VulkanSimPipelineSet::SimFragPushConstants>{simBuffersPushConstants});
            cmdBuffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics,
                    *global.pipelines.fullscreenPressure.layout,
                    0,
                    {*simFrameData.simBuffersDescriptorSet},
                    {});
            cmdBuffer.draw(6, 1, 0, 0);
            cmdBuffer.endRenderPass();
        }

        {
            auto imguiRenderPassInfo = vk::RenderPassBeginInfo();
            imguiRenderPassInfo.renderPass = global.imguiRenderPass;
            imguiRenderPassInfo.framebuffer = **swImageData.framebuffer;
            imguiRenderPassInfo.renderArea = global.imguiRenderArea;
            imguiRenderPassInfo.clearValueCount = 1;
            imguiRenderPassInfo.pClearValues = &clearColor;
            cmdBuffer.beginRenderPass(imguiRenderPassInfo, vk::SubpassContents::eInline);


            {
                ImDrawData *draw_data = ImGui::GetDrawData();
                ImGui_ImplVulkan_RenderDrawData(draw_data, global.imguiContext, cmdBuffer, global.imguiRenderPass, VK_SAMPLE_COUNT_1_BIT);
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