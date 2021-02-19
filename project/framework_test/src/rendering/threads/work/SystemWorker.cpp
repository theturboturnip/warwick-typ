//
// Created by samuel on 13/02/2021.
//

#include "SystemWorker.h"

#include "rendering/shaders/global_structures.h"

std::array<const char*, 5> SystemWorker::scalarQuantity = {
        "None",
        "Velocity X [TODO]",
        "Velocity Y [TODO]",
        "Pressure [TODO]",
        "Vorticity [TODO]",
};
std::array<const char*, 2> SystemWorker::vectorQuantity = {
        "None",
        "Velocity [TODO]",
};
std::array<const char*, 4> SystemWorker::particleTrailType = {
        "None",
        "Streakline [TODO]",
        "Pathline [TODO]",
        "Ribbon [TODO]",
};

SystemWorker::SystemWorker(VulkanSimAppData &data)
        : data(data),
          global(data.globalData)
{
}


SystemWorkerOut SystemWorker::work(SystemWorkerIn input) {
    // Handle SDL inputs
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

        if (input.perf.elapsedSimTime > 0.0) {
            ImGui::Text("Elapsed Sim Time: %.2f", input.perf.elapsedSimTime);
            ImGui::Text("Elapsed Real Time for Sim: %.2f", input.perf.elapsedRealTimeWhileSimWanted);
            double simRealTimeRatio = input.perf.elapsedSimTime / input.perf.elapsedRealTimeWhileSimWanted;
            ImGui::Text("Sim/Real Time Ratio: %.2f", simRealTimeRatio);

            if (input.perf.simFrameNum >= input.perf.simFrameTimes.size()) {
                // Note that this is real-time, not sim-time.
                float sumSimFrameTimes = std::accumulate(input.perf.simFrameTimes.begin(), input.perf.simFrameTimes.end(), 0.0f);
                float avgSimFrameTime = sumSimFrameTimes / input.perf.simFrameTimes.size();
                ImGui::Text("Avg Sim frame real-time: %.2fms", avgSimFrameTime*1000.0f);

                float sumSimTickLength = std::accumulate(input.perf.simTickLengths.begin(), input.perf.simTickLengths.end(), 0.0f);
                float avgSimTickLength = sumSimTickLength / input.perf.simTickLengths.size();
                ImGui::Text("Avg Sim frame sim-time: %.2fms", avgSimTickLength*1000.0f);
            } else {
                ImGui::Text("Not enough frames for Sim FPS");
            }
        }
    }
    ImGui::End();

    ImGui::Begin("Visualization", nullptr, 0);
    {
        ImGui::Text("Scalar Quantity");
        // From https://github.com/ocornut/imgui/issues/1658
        if (ImGui::BeginCombo("##Scalar Quantity", scalarQuantity[(int)vizScalar])) {
            for (size_t i = 0; i < scalarQuantity.size(); i++) {
                bool is_selected = (i == (size_t)vizScalar);
                if (ImGui::Selectable(scalarQuantity[i], is_selected)) {
                    vizScalar = (ScalarQuantity)i;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (vizScalar != ScalarQuantity::None)
            showRange(&vizScalarRange);

        ImGui::NewLine();
        ImGui::Text("Vector Quantity");
        if (ImGui::BeginCombo("##Vector Quantity", vectorQuantity[(int)vizVector])) {
            for (size_t i = 0; i < vectorQuantity.size(); i++) {
                bool is_selected = (i == (size_t)vizVector);
                if (ImGui::Selectable(vectorQuantity[i], is_selected)) {
                    vizVector = (VectorQuantity)i;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (vizVector != VectorQuantity::None)
            showRange(&vizVectorMagnitudeRange);

        ImGui::NewLine();
        ImGui::Checkbox("Streamline Overlay", &overlayStreamlines);

        ImGui::NewLine();
        ImGui::Checkbox("Particles", &showParticles);
        if (showParticles) {
            ImGui::Indent();

            ImGui::Checkbox("Render as Glyphs", &renderParticleGlyphs);

            ImGui::NewLine();
            ImGui::Text("Particle Trail Type");
            if (ImGui::BeginCombo("##TraceType", particleTrailType[(int)trailType])) {
                for (size_t i = 0; i < particleTrailType.size(); i++) {
                    bool is_selected = (i == (size_t)trailType);
                    if (ImGui::Selectable(particleTrailType[i], is_selected)) {
                        trailType = (ParticleTrailType)i;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Unindent();
        }
    }
    ImGui::End();

    //ImGui::SetNextWindowSize(ImVec2(simSize.pixel_size.x+2, simSize.pixel_size.y+2));
    ImGui::Begin("Simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Image(
            (ImTextureID)*simFrameData.vizFramebufferDescriptorSet,
            ImVec2(global.simSize.padded_pixel_size.x*1.5, global.simSize.padded_pixel_size.y*1.5)
    );
    ImGui::End();
    ImGui::Render();

    auto simBuffersPushConstants = Shaders::SimDataBufferStats{
        .sim_pixelWidth=global.simSize.padded_pixel_size.x,
        .sim_pixelHeight=global.simSize.padded_pixel_size.y,
        .sim_columnStride=global.simSize.padded_pixel_size.y,
        .sim_totalPixels=(uint32_t)global.simSize.pixel_count()
    };

    const auto& graphicsCmdBuffer = *simFrameData.threadOutputs.graphicsCommandBuffer;
    const auto& computeCmdBuffer = *simFrameData.threadOutputs.computeCommandBuffer;

    {
        computeCmdBuffer.reset({});

        auto beginInfo = vk::CommandBufferBeginInfo();
        computeCmdBuffer.begin(beginInfo);

        {
            // Transfer the simBuffersImage to eGeneral so it can be written next frame.
            transferImageLayout(
                    computeCmdBuffer,
                    *simFrameData.simBuffersImage,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits(0), vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader
            );
        }

        {
            // Run the compute shader
            computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *global.pipelines.computeSimDataImage);
            computeCmdBuffer.pushConstants(
                                        *global.pipelines.computeSimDataImage.layout,
                                        vk::ShaderStageFlagBits::eCompute,
                                        0,
                                        vk::ArrayProxy<const VulkanSimPipelineSet::SimFragPushConstants>{simBuffersPushConstants});
            computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                *global.pipelines.computeSimDataImage.layout,
                                                0,
                                                {*simFrameData.simBuffersComputeDescriptorSet},
                                                {});
            // Group size of 16 -> group count in each direction is size/16
            // If size isn't a multiple of 16, get extra group to cover the remainder
            // i.e. (size + 15) / 16
            //  because if size % 16 == 0 then (size / 16) === (size + 15)/16
            //  otherwise (size + 15)/16 === (size / 16) + 1
            computeCmdBuffer.dispatch((simFrameData.simBuffersImage.size.x + 15)/16, (simFrameData.simBuffersImage.size.y+15)/16, 1);
        }
        computeCmdBuffer.end();
    }

    {
        graphicsCmdBuffer.reset({});

        auto clearColor = vk::ClearValue(vk::ClearColorValue());
        clearColor.color.setFloat32({0.0f, 0.0f, 0.1f, 1.0f});

        auto beginInfo = vk::CommandBufferBeginInfo();
        graphicsCmdBuffer.begin(beginInfo);

        {
            // Transfer the simBuffersImage layout so that the shader can read it
            transferImageLayout(
                    graphicsCmdBuffer,
                    *simFrameData.simBuffersImage,
                    vk::ImageLayout::eGeneral,vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::AccessFlagBits(0), vk::AccessFlagBits::eShaderRead,
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader
            );
        }

        {
            auto simRenderPassInfo = vk::RenderPassBeginInfo();
            simRenderPassInfo.renderPass = global.vizRenderPass;
            simRenderPassInfo.framebuffer = *simFrameData.vizFramebuffer;
            simRenderPassInfo.renderArea = global.vizRect;
            simRenderPassInfo.clearValueCount = 1;
            simRenderPassInfo.pClearValues = &clearColor;
            graphicsCmdBuffer.beginRenderPass(simRenderPassInfo, vk::SubpassContents::eInline);
            graphicsCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *global.pipelines.fullscreenPressure);
            graphicsCmdBuffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics,
                    *global.pipelines.fullscreenPressure.layout,
                    0,
                    {*simFrameData.simBuffersImageDescriptorSet},
                    {});
            graphicsCmdBuffer.draw(6, 1, 0, 0);
            graphicsCmdBuffer.endRenderPass();
        }

        {
            auto imguiRenderPassInfo = vk::RenderPassBeginInfo();
            imguiRenderPassInfo.renderPass = global.finalCompositeRenderPass;
            imguiRenderPassInfo.framebuffer = **swImageData.framebuffer;
            imguiRenderPassInfo.renderArea = global.finalCompositeRect;
            imguiRenderPassInfo.clearValueCount = 1;
            imguiRenderPassInfo.pClearValues = &clearColor;
            graphicsCmdBuffer.beginRenderPass(imguiRenderPassInfo, vk::SubpassContents::eInline);


            {
                ImDrawData *draw_data = ImGui::GetDrawData();
                ImGui_ImplVulkan_RenderDrawData(draw_data, global.imguiContext, graphicsCmdBuffer, global.finalCompositeRenderPass, VK_SAMPLE_COUNT_1_BIT);
            }

            graphicsCmdBuffer.endRenderPass();
        }
        graphicsCmdBuffer.end();
    }

    return SystemWorkerOut{
            .wantsQuit = wantsQuit,
            .wantsRunSim = wantsRunSim,
            .graphicsCmdBuffer = graphicsCmdBuffer,
            .computeCmdBuffer = computeCmdBuffer,
    };
}

void
SystemWorker::transferImageLayout(vk::CommandBuffer cmdBuffer, vk::Image image,
                                  vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                  vk::AccessFlags oldAccess, vk::AccessFlags newAccess,
                                  vk::PipelineStageFlags oldStage, vk::PipelineStageFlags newStage) {
    auto imageBarrier = vk::ImageMemoryBarrier{};
    imageBarrier.image = image;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    // We don't do any mipmapping/texture arrays ever - only use the first mip level, and the first array layer
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    imageBarrier.srcAccessMask = oldAccess;
    imageBarrier.dstAccessMask = newAccess;

    cmdBuffer.pipelineBarrier(
        oldStage, newStage, vk::DependencyFlagBits(0), {}, {}, {imageBarrier}
    );
}

void SystemWorker::showRange(VizValueRange* range) {
    ImGui::PushID(range);
    ImGui::Checkbox("Auto-Range", &range->autoRange);
    if (!range->autoRange) {
        float rangeValue[2] = {range->min, range->max};
        ImGui::InputFloat2("Range", static_cast<float*>(rangeValue));
        range->min = rangeValue[0];
        range->max = rangeValue[1];
    }
    ImGui::PopID();
}
