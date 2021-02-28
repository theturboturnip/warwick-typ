//
// Created by samuel on 13/02/2021.
//

#include <rendering/vulkan/viz/vertex.h>
#include "SystemWorker.h"

#include "rendering/shaders/global_structures.h"

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

    bool shouldResetParticles = false;

    ImGui::Begin("Visualization", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        ImGui::Text("Scalar Quantity");
        // From https://github.com/ocornut/imgui/issues/1658
        if (ImGui::BeginCombo("##Scalar Quantity", scalarQuantityStrs[(int)vizScalar])) {
            for (size_t i = 0; i < scalarQuantityStrs.size(); i++) {
                bool is_selected = (i == (size_t)vizScalar);
                if (ImGui::Selectable(scalarQuantityStrs[i], is_selected)) {
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
        if (ImGui::BeginCombo("##Vector Quantity", vectorQuantityStrs[(int)vizVector])) {
            for (size_t i = 0; i < vectorQuantityStrs.size(); i++) {
                bool is_selected = (i == (size_t)vizVector);
                if (ImGui::Selectable(vectorQuantityStrs[i], is_selected)) {
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
        ImGui::InputFloat2("Grid Spacing", vizVectorSpacing);
        ImGui::SliderFloat("Vector Size", &vizVectorSize, 0.01, 0.1, "%.5f", 2.0f);


//        ImGui::NewLine();
//        ImGui::Checkbox("Streamline Overlay", &overlayStreamlines);

        ImGui::NewLine();
        if (ImGui::Checkbox("Particles", &simulateParticles)) {
            // Particles have been toggled
            if (!simulateParticles) {
                shouldResetParticles = true;
                fprintf(stdout, "resetting particles\n");
            }
        }
        if (simulateParticles) {
            ImGui::Indent();

            ImGui::Checkbox("Render as Glyphs", &renderParticleGlyphs);
            if (renderParticleGlyphs) {
                ImGui::SliderFloat("Size", &particleGlyphSize, 0.01, 0.1, "%.5f", 2.0f);
            }

            ImGui::SliderFloat("Spawn Speed", &particleSpawnFreq, 1, 100, "%.1f", 2.0f);

            ImGui::Checkbox("Lock to Sim", &lockParticleToSimulation);
            if (!lockParticleToSimulation) {
                ImGui::Indent();
                ImGui::InputFloat("Hz Timestep", &particleUnlockedSimFreq);
                ImGui::Unindent();
            }

            ImGui::Unindent();
        }
    }
    ImGui::End();

    //ImGui::SetNextWindowSize(ImVec2(simSize.pixel_size.x+2, simSize.pixel_size.y+2));
    ImGui::Begin("Simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Image(
            (ImTextureID)*data.sharedFrameData.vizFramebufferDescriptorSet,
            ImVec2(global.simSize.padded_pixel_size.x*2, global.simSize.padded_pixel_size.y*2)
    );
    ImGui::End();
    ImGui::Render();

    auto& simFrameData = data.perFrameData[input.simFrameIndex];

    // Get a new swapchain image
    uint32_t swapchainImageIdx = 0;
    global.device.acquireNextImageKHR(
            *global.swapchain,
            UINT64_MAX,
            *simFrameData.imageAcquiredCanRender, nullptr,
            &swapchainImageIdx
    );
    auto& swImageData = data.swapchainImageData[swapchainImageIdx];
//        fprintf(stderr, "told to get swapchain image %d\n", swapchainImageIdx);


    bool spawnNewParticleThisTick = false;
    particleSpawnTimer += input.thisSimTickLength;
    if (particleSpawnTimer > (1.0f / particleSpawnFreq)) {
        spawnNewParticleThisTick = true;
        particleSpawnTimer = 0;
    }

    // Wait for this frame's command buffers to become available
//    fprintf(stderr, "Waiting for cmdbuffer fence\n");
    // TODO waiting for swapchain image data is unnecessary
    if (swImageData.inFlight != (vk::Fence)nullptr && swImageData.inFlight != *simFrameData.frameCmdBuffersInUse) {
        global.device.waitForFences({*simFrameData.frameCmdBuffersInUse, swImageData.inFlight}, true, UINT64_MAX);
    } else {
        global.device.waitForFences({*simFrameData.frameCmdBuffersInUse}, true, UINT64_MAX);
    }
    swImageData.inFlight = *simFrameData.frameCmdBuffersInUse;

//        fprintf(stderr, "got sim frame and swapchain image\n");

    auto simBuffersPushConstants = Shaders::SimDataBufferStats{
        .sim_pixelWidth=global.simSize.padded_pixel_size.x,
        .sim_pixelHeight=global.simSize.padded_pixel_size.y,
        .sim_columnStride=global.simSize.padded_pixel_size.y,
        .sim_totalPixels=(uint32_t)global.simSize.pixel_count()
    };

    const auto& computeCmdBuffer = *simFrameData.computeCmdBuffer;
    const auto& graphicsCmdBuffer = *simFrameData.renderCmdBuffer;

    {
        computeCmdBuffer.reset({});

        auto beginInfo = vk::CommandBufferBeginInfo();
        computeCmdBuffer.begin(beginInfo);

        {
            // Transfer the simBuffersImage to eGeneral so it can be written next frame.
            transferImageLayout(
                    computeCmdBuffer,
                    *data.sharedFrameData.simDataImage,
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
                                        vk::ArrayProxy<const Shaders::SimDataBufferStats>{simBuffersPushConstants});
            computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                *global.pipelines.computeSimDataImage.layout,
                                                0,
                                                vk::ArrayProxy<const vk::DescriptorSet>{
                                                    *simFrameData.simBufferCopyInput_comp_ds,
                                                    *data.sharedFrameData.simBufferCopyOutput_comp_ds
                                                },
                                                {});
            // Group size of 16 -> group count in each direction is size/16
            // If size isn't a multiple of 16, get extra group to cover the remainder
            // i.e. (size + 15) / 16
            //  because if size % 16 == 0 then (size / 16) === (size + 15)/16
            //  otherwise (size + 15)/16 === (size / 16) + 1
            computeCmdBuffer.dispatch((global.simSize.padded_pixel_size.x*2 + 15)/16, (global.simSize.padded_pixel_size.y*2+15)/16, 1);
        }
        // Beore the image layouer is transferred, use it for min/max
        if (vizScalar != ScalarQuantity::None) {
            // Perform scalar extraction

            // Make the quantityScalar image writable
            transferImageLayout(
                    computeCmdBuffer,
                    *data.sharedFrameData.quantityScalar,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits(0), vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader
            );

            auto scalarExtractParams = Shaders::ScalarExtractParams{
                    .simDataImage_width = data.sharedFrameData.simDataImage.size.x,
                    .simDataImage_height = data.sharedFrameData.simDataImage.size.y
            };

            // Run the compute shader
            computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *global.pipelines.computeScalarExtract[vizScalar]);
            computeCmdBuffer.pushConstants(
                    *global.pipelines.computeScalarExtract[vizScalar].layout,
                    vk::ShaderStageFlagBits::eCompute,
                    0,
                    vk::ArrayProxy<const Shaders::ScalarExtractParams>{scalarExtractParams});
            computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                *global.pipelines.computeScalarExtract[vizScalar].layout,
                                                0,
                                                vk::ArrayProxy<const vk::DescriptorSet>{
                                                        *data.sharedFrameData.simBufferCopyOutput_comp_ds,

                                                        *data.sharedFrameData.quantityScalar_comp_ds,
                                                        *data.sharedFrameData.quantityScalarReducer.getInputDescriptorSets().buffer_comp_ds
                                                },
                                                {});
            // Group size of 16 -> group count in each direction is size/16
            // If size isn't a multiple of 16, get extra group to cover the remainder
            // i.e. (size + 15) / 16
            //  because if size % 16 == 0 then (size / 16) === (size + 15)/16
            //  otherwise (size + 15)/16 === (size / 16) + 1
            computeCmdBuffer.dispatch((data.sharedFrameData.quantityScalar.size.x + 15)/16, (data.sharedFrameData.quantityScalar.size.y+15)/16, 1);

            if (vizScalarRange.autoRange) {
                // Perform the reduction, copying the data into quantityScalar_range
                data.sharedFrameData.quantityScalarReducer.enqueueReductionFromInput(computeCmdBuffer, simFrameData.quantityScalar_range.getGpuBuffer());
            } else {
                // Copy quantity ranges in from CPU
                {
                    {
                        auto memory = simFrameData.quantityScalar_range.mapCPUMemory(*global.context.device);
                        auto* quantityScalar_range = (Shaders::FloatRange*)(*memory);
                        *quantityScalar_range = Shaders::FloatRange{
                                vizScalarRange.min, vizScalarRange.max
                        };

                        // Auto unmapped
                    }
                    simFrameData.quantityScalar_range.scheduleCopyToGPU(computeCmdBuffer);
                }

            }

            // Make the quantityScalar image readable
            transferImageLayout(
                    computeCmdBuffer,
                    *data.sharedFrameData.quantityScalar,
                    vk::ImageLayout::eGeneral,vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader
            );
        }
        if (vizVector != VectorQuantity::None) {
            // Perform vector extraction

            // Make the quantityVector image writable
            transferImageLayout(
                    computeCmdBuffer,
                    *data.sharedFrameData.quantityVector,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits(0), vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader
            );

            auto vectorExtractParams = Shaders::VectorExtractParams{
                    .simDataImage_width = data.sharedFrameData.simDataImage.size.x,
                    .simDataImage_height = data.sharedFrameData.simDataImage.size.y
            };

            // Run the compute shader
            computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *global.pipelines.computeVectorExtract[vizVector]);
            computeCmdBuffer.pushConstants(
                    *global.pipelines.computeVectorExtract[vizVector].layout,
                    vk::ShaderStageFlagBits::eCompute,
                    0,
                    vk::ArrayProxy<const Shaders::VectorExtractParams>{vectorExtractParams});
            computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                *global.pipelines.computeVectorExtract[vizVector].layout,
                                                0,
                                                vk::ArrayProxy<const vk::DescriptorSet>{
                                                        *data.sharedFrameData.simBufferCopyOutput_comp_ds,

                                                        *data.sharedFrameData.quantityVector_comp_ds,
                                                        *data.sharedFrameData.quantityVectorReducer.getInputDescriptorSets().buffer_comp_ds
                                                },
                                                {});
            // Group size of 16 -> group count in each direction is size/16
            // If size isn't a multiple of 16, get extra group to cover the remainder
            // i.e. (size + 15) / 16
            //  because if size % 16 == 0 then (size / 16) === (size + 15)/16
            //  otherwise (size + 15)/16 === (size / 16) + 1
            computeCmdBuffer.dispatch((data.sharedFrameData.quantityVector.size.x + 15)/16, (data.sharedFrameData.quantityVector.size.y+15)/16, 1);

            if (vizVectorMagnitudeRange.autoRange) {
                // Perform the reduction, copying the data into quantityVector_range
                data.sharedFrameData.quantityVectorReducer.enqueueReductionFromInput(computeCmdBuffer, simFrameData.quantityVector_range.getGpuBuffer());
            } else {
                // Copy quantity ranges in from CPU
                {
                    {
                        auto memory = simFrameData.quantityVector_range.mapCPUMemory(*global.context.device);
                        auto* quantityVector_range = (Shaders::FloatRange*)(*memory);
                        *quantityVector_range = Shaders::FloatRange{
                                vizVectorMagnitudeRange.min, vizVectorMagnitudeRange.max
                        };

                        // Auto unmapped
                    }
                    simFrameData.quantityVector_range.scheduleCopyToGPU(computeCmdBuffer);
                }

            }

            // Make the quantityVector image readable
            transferImageLayout(
                    computeCmdBuffer,
                    *data.sharedFrameData.quantityVector,
                    vk::ImageLayout::eGeneral,vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader
            );

            // Generate vector arrow instances
            {
                // Zero-out the vectorArrowInstanceData,
                computeCmdBuffer.fillBuffer(*data.sharedFrameData.vectorArrowInstanceData, 0, data.sharedFrameData.vectorArrowInstanceData.size, 0);
                // reset the quantityVectorIndirectDrawData.
                // The CPU-side buffer already has the correct data, we just need to transfer it in.
                data.sharedFrameData.quantityVectorIndirectDrawData.scheduleCopyToGPU(computeCmdBuffer);

                // Wait for transfers to finish before starting the compute shader
                // (the buffer fill also counts as a transfer)
                fullMemoryBarrier(
                    computeCmdBuffer,
                    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                    vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead
                );

                // Run the compute shader
                auto vectorGenerateParams = Shaders::VectorArrowGenerateParams{
                    .gridCount_x = (uint32_t)(1.0 / vizVectorSpacing[0]) + 1,
                    .gridCount_y = (uint32_t)(1.0 / vizVectorSpacing[1]) + 1,
                    .baseScale = vizVectorSize,
                    .render_heightDivWidth = global.vizRect.extent.height * 1.0f / global.vizRect.extent.width
                };
                computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *global.pipelines.computeVectorArrowGenerate);
                computeCmdBuffer.pushConstants(
                        *global.pipelines.computeVectorArrowGenerate.layout,
                        vk::ShaderStageFlagBits::eCompute,
                        0,
                        vk::ArrayProxy<const Shaders::VectorArrowGenerateParams>{vectorGenerateParams});
                computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                    *global.pipelines.computeVectorArrowGenerate.layout,
                                                    0,
                                                    vk::ArrayProxy<const vk::DescriptorSet>{
                                                            *data.sharedFrameData.quantityVectorSampler_comp_ds,
                                                            *simFrameData.quantityVector_range_comp_ds,

                                                            *data.sharedFrameData.vectorArrowInstanceData_comp_ds,
                                                            *data.sharedFrameData.quantityVectorIndirectDrawData_comp_ds
                                                    },
                                                    {});
                // Group size of 16 -> group count in each direction is size/16
                // If size isn't a multiple of 16, get extra group to cover the remainder
                // i.e. (size + 15) / 16
                //  because if size % 16 == 0 then (size / 16) === (size + 15)/16
                //  otherwise (size + 15)/16 === (size / 16) + 1
                computeCmdBuffer.dispatch((vectorGenerateParams.gridCount_x + 15)/16, (vectorGenerateParams.gridCount_y+15)/16, 1);

            }
        }

        {
            // Transfer the simBuffersImage layout so that the next shaders can read it
            // Make ShaderWrites to a General-layout image in the ComputeShader
            //   available + visible to ShaderReads from a ShaderReadOnlyOptimal-layout image in the ComputeShader.
            // Even if we're not updating the particles this frame, transitioning the layout is required.
            transferImageLayout(
                    computeCmdBuffer,
                    *data.sharedFrameData.simDataImage,
                    vk::ImageLayout::eGeneral,vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader
            );
        }

        // Particle Update
        // If we're not locking the particle sim to the actual sim, always run the particle sim.
        // If we are locking the sims together, only run the particle sim if the input ran the sim.
        if (simulateParticles && (!lockParticleToSimulation || input.shouldSimParticles)) {
            // Copy index draw list -> index simulate list for the emit shader to add to
            {
                auto copy = vk::BufferCopy{};
                copy.srcOffset = 0;
                copy.dstOffset = 0;
                copy.size = data.sharedFrameData.particleIndexDrawList.size;
                computeCmdBuffer.copyBuffer(*data.sharedFrameData.particleIndexDrawList,
                                            *data.sharedFrameData.particleIndexSimulateList, {copy});
            }

            // Setup the emitter points
            {
                {
                    DASSERT(global.props.maxParicleEmitters >= 6);
                    auto memory = simFrameData.particleEmitters.mapCPUMemory(*global.context.device);
                    auto* emitterData = (Shaders::ParticleEmitter*)(*memory);
                    emitterData[0] = Shaders::ParticleEmitter {
                        .position = glm::vec4(0, 0.1, 0, 0),
                        .color = glm::vec4(1, 0, 0, 1)
                    };
                    emitterData[1] = Shaders::ParticleEmitter {
                        .position = glm::vec4(0, 0.4, 0, 0),
                        .color = glm::vec4(0, 1, 0, 1)
                    };
                    emitterData[2] = Shaders::ParticleEmitter {
                        .position = glm::vec4(0, 0.6, 0, 0),
                        .color = glm::vec4(0, 0, 1, 1)
                    };
                    emitterData[3] = Shaders::ParticleEmitter {
                        .position = glm::vec4(0, 0.9, 0, 0),
                        .color = glm::vec4(1, 1, 1, 1)
                    };
                    emitterData[4] = Shaders::ParticleEmitter {
                            .position = glm::vec4(0, 0.3, 0, 0),
                            .color = glm::vec4(0, 0, 1, 1)
                    };
                    emitterData[5] = Shaders::ParticleEmitter {
                            .position = glm::vec4(0, 0.7, 0, 0),
                            .color = glm::vec4(1, 1, 1, 1)
                    };

                    // Auto unmapped
                }
                simFrameData.particleEmitters.scheduleCopyToGPU(computeCmdBuffer);
            }

            // Run the kickoff shader
            {
                // Initial pipeline barrier
                // particleIndexDrawList is in input.
                //      was written last frame, multiple semaphores inbetween => writes are available and visible
                // particlesToEmit is an output, we don't care what's in there
                // particleIndirectCommands is an output, we don't care

                auto particleKickoff = Shaders::ParticleKickoffParams{
                        .emitterCount = (spawnNewParticleThisTick ? 6u : 0u)
                };
                computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                              *global.pipelines.computeParticleKickoff);
                computeCmdBuffer.pushConstants(
                        *global.pipelines.computeParticleKickoff.layout,
                        vk::ShaderStageFlagBits::eCompute,
                        0,
                        vk::ArrayProxy<const Shaders::ParticleKickoffParams>{particleKickoff});
                computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                    *global.pipelines.computeParticleKickoff.layout,
                                                    0,
                                                    {
                                                        // Input
                                                        *data.sharedFrameData.particleIndexDrawList_comp_ds, //particlesDrawnLastFrame

                                                        // Output
                                                        *data.sharedFrameData.particlesToEmit_comp_ds, //particlesToEmit
                                                        *data.sharedFrameData.particleIndirectCommands_comp_ds, //indirectCmds
                                                    },
                                                    {});
                computeCmdBuffer.dispatch(1, 1, 1);
            }

            // Run the emit shader
            {
                // particleEmitters, particleIndexSimulate list are inputs, need to wait for the transfers to finish
                // particlesToEmit is an input, need to wait for the previous compute to finish
                // Reads from particleIndirectCommands, need to wait for previous compute to finish BEFORE DrawIndirectBit

                // So full-memory barrier from Compute->DrawIndirect
                // and full-memory barrier from Transfer->Compute?

                // Make ShaderWrites from the ComputeShader stage available + visible to IndirectCommandReads in the DrawIndirect stage
                fullMemoryBarrier(computeCmdBuffer,
                                  vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eDrawIndirect,
                                  vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eIndirectCommandRead);
                // Make TransferWrites from the Transfer stage available + visible to the ShaderReads in the ComputeShader phase.
                fullMemoryBarrier(computeCmdBuffer,
                                  vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                  vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);

                computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                              *global.pipelines.computeParticleEmit);
                computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                    *global.pipelines.computeParticleEmit.layout,
                                                    0,
                                                    {
                                                        // Input
                                                        *simFrameData.particleEmitters_comp_ds, //emitters
                                                        *data.sharedFrameData.particlesToEmit_comp_ds, //particlesToEmit

                                                        // Input/output
                                                        *data.sharedFrameData.particleIndexSimulateList_comp_ds, //particlesToSimIndexList
                                                        *data.sharedFrameData.inactiveParticleIndexList_comp_ds, //inactiveParticleIndexList

                                                        // Output
                                                        *data.sharedFrameData.particleDataArray_comp_ds, //particleDatas
                                                    },
                                                    {});
                computeCmdBuffer.dispatchIndirect(
                    *data.sharedFrameData.particleIndirectCommands,
                    offsetof(Shaders::ParticleIndirectCommands, particleEmitCmd)
                );
            }

            // Run the simulate shader
            {
                // particleIndexSimulateList is an input from the emit shader.
                // the indirect draw is an input from the kickoff shader, but the emit shader has already waited so we don't have to.

                // Make ShaderWrites from the ComputeShader stage available + visible to the ShaderReads in the ComputeShader phase.
                fullMemoryBarrier(computeCmdBuffer,
                                  vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                                  vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

                auto particleSimParams = Shaders::ParticleSimulateParams{
                        .timestep = lockParticleToSimulation ? input.thisSimTickLength : (1.0f/particleUnlockedSimFreq),
                        .xLength = global.simSize.physical_size.x,
                        .yLength = global.simSize.physical_size.y
                };
                computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                              *global.pipelines.computeParticleSimulate);
                computeCmdBuffer.pushConstants(
                        *global.pipelines.computeParticleSimulate.layout,
                        vk::ShaderStageFlagBits::eCompute,
                        0,
                        vk::ArrayProxy<const Shaders::ParticleSimulateParams>{particleSimParams});
                computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                                    *global.pipelines.computeParticleSimulate.layout,
                                                    0,
                                                    {
                                                        // Inputs
                                                        *data.sharedFrameData.particleIndexSimulateList_comp_ds, //particlesToSimIndexList
                                                        *data.sharedFrameData.simDataSampler_comp_ds,    //simBufferDataSampler

                                                        // Input/Outputs
                                                        *data.sharedFrameData.particleIndexDrawList_comp_ds, //particlesToDrawIndexList
                                                        *data.sharedFrameData.inactiveParticleIndexList_comp_ds, //inactiveParticleIndexList

                                                        // Outputs
                                                        *data.sharedFrameData.particleIndirectCommands_comp_ds, //indirectCmds
                                                        *data.sharedFrameData.particleDataArray_comp_ds, //particleDatas
                                                    },
                                                    {});
                computeCmdBuffer.dispatchIndirect(
                    *data.sharedFrameData.particleIndirectCommands,
                    offsetof(Shaders::ParticleIndirectCommands, particleSimCmd)
                );

                // Don't need to sync writes to the particleIndirectCommands/particleDataArray
                // because that's done on the graphics pipeline after a semaphore.
            }
        } else if (shouldResetParticles) {
            // Reset the particles:
            // Zero out the "to draw list", which is used as a base for the simulation normally
            computeCmdBuffer.fillBuffer(*data.sharedFrameData.particleIndexDrawList, 0, data.sharedFrameData.particleIndexDrawList.size, 0);
            // Copy the reset data into the inactiveParticleIndexList
            auto copy = vk::BufferCopy{};
            copy.srcOffset = 0;
            copy.dstOffset = 0;
            copy.size = data.sharedFrameData.inactiveParticleIndexList.size;
            computeCmdBuffer.copyBuffer(
                data.sharedFrameData.inactiveParticleIndexList_resetData.getGpuBuffer(),
                data.sharedFrameData.inactiveParticleIndexList.getGpuBuffer(),
                {copy}
            );
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
            auto simRenderPassInfo = vk::RenderPassBeginInfo();
            simRenderPassInfo.renderPass = global.vizRenderPass;
            simRenderPassInfo.framebuffer = *data.sharedFrameData.vizFramebuffer;
            simRenderPassInfo.renderArea = global.vizRect;
            simRenderPassInfo.clearValueCount = 1;
            simRenderPassInfo.pClearValues = &clearColor;
            graphicsCmdBuffer.beginRenderPass(simRenderPassInfo, vk::SubpassContents::eInline);

            // Render background
            {
                // Colors are ABGR
                auto quantityScalar_pushConsts = Shaders::QuantityScalarParams{
                    .colorRange32Bit = {
                            glm::packUnorm4x8({0, 0, 1, 1}), // Blue TODO this is only here cuz idk what else could go here
                            glm::packUnorm4x8({0, 0.5, 1, 1}), // Blue
                            glm::packUnorm4x8({0, 0.5, 0.5, 1}), // 1/2 between blue, turquoise
                            glm::packUnorm4x8({0, 1, 1, 1}), // Turquoise
                            glm::packUnorm4x8({0, 1, 0, 1}), // Green
                            glm::packUnorm4x8({1, 1, 0, 1}), // Yellow
                            glm::packUnorm4x8({1, 0.5, 0, 1}), // Orange
                            glm::packUnorm4x8({1, 0, 0, 1}), // Red
                    },
                    .fluidColor32Bit = glm::packUnorm4x8({0.5, 0.5, 0.5, 1.0}),
                    .obstacleColor32Bit = glm::packUnorm4x8({0, 0, 0, 1})
                };

                const auto &quantityScalar = global.pipelines.quantityScalar[vizScalar];
                graphicsCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *quantityScalar);
                graphicsCmdBuffer.pushConstants(
                        *quantityScalar.layout,
                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                        0,
                        vk::ArrayProxy<const Shaders::QuantityScalarParams>{quantityScalar_pushConsts}
                );
                graphicsCmdBuffer.bindDescriptorSets(
                        vk::PipelineBindPoint::eGraphics,
                        *quantityScalar.layout,
                        0,
                        {
                            *data.sharedFrameData.quantityScalarSampler_frag_ds,
                            *simFrameData.quantityScalar_range_frag_ds,
                        },
                        {});
                graphicsCmdBuffer.draw(4, 1, 0, 0);
            }

            if (vizVector != VectorQuantity::None) {
                auto vectorArrowPushConsts = Shaders::InstancedVectorArrowParams{
                        .dummy = 1
                };

                graphicsCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *global.pipelines.vectorArrow);
                graphicsCmdBuffer.pushConstants(
                        *global.pipelines.vectorArrow.layout,
                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                        0,
                        vk::ArrayProxy<const Shaders::InstancedVectorArrowParams>{vectorArrowPushConsts}
                );
                graphicsCmdBuffer.bindVertexBuffers(0, {data.sharedFrameData.vectorArrowVertexIndexData.getGpuBuffer()}, {0});
                graphicsCmdBuffer.bindIndexBuffer(data.sharedFrameData.vectorArrowVertexIndexData.getGpuBuffer(), sizeof(Vertex) * 7, vk::IndexType::eUint16);
                graphicsCmdBuffer.bindDescriptorSets(
                        vk::PipelineBindPoint::eGraphics,
                        *global.pipelines.vectorArrow.layout,
                        0,
                        {
                            *data.sharedFrameData.vectorArrowInstanceData_vert_ds
                        },
                        {});
                graphicsCmdBuffer.drawIndexedIndirect(
                        data.sharedFrameData.quantityVectorIndirectDrawData.getGpuBuffer(),
                        offsetof(Shaders::VectorArrowIndirectCommands, vectorArrowDrawCmd),
                        1, 0
                );
            }

            if (simulateParticles && renderParticleGlyphs){
                auto particlePushConsts = Shaders::InstancedParticleParams{
                        .baseScale = particleGlyphSize,
                        .render_heightDivWidth = global.vizRect.extent.height * 1.0f / global.vizRect.extent.width
                };

                graphicsCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *global.pipelines.particle);
                graphicsCmdBuffer.pushConstants(
                        *global.pipelines.particle.layout,
                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                        0,
                        vk::ArrayProxy<const Shaders::InstancedParticleParams>{particlePushConsts}
                );
                graphicsCmdBuffer.bindVertexBuffers(0, {data.sharedFrameData.particleVertexData.getGpuBuffer()}, {0});
                graphicsCmdBuffer.bindDescriptorSets(
                        vk::PipelineBindPoint::eGraphics,
                        *global.pipelines.particle.layout,
                        0,
                        {
                                *data.sharedFrameData.particleIndexDrawList_vert_ds, // particlesToDrawIndexList
                                *data.sharedFrameData.particleDataArray_vert_ds // particleDatas
                        },
                        {});
                graphicsCmdBuffer.drawIndirect(
                        *data.sharedFrameData.particleIndirectCommands,
                        offsetof(Shaders::ParticleIndirectCommands, particleDrawCmd),
                        1, 0
                );
            }

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

    // Submit the command buffers
    // Send the compute work
    {
        vk::SubmitInfo submitInfo{};
        std::vector<vk::Semaphore> waitSemaphores = {*simFrameData.simFinishedCanCompute};
        std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eComputeShader};
        if (input.perf.currentFrameNum > 0) {
            // Find previous frame idx, use to check that the previous render has finished
            // Do (i + size - 1) % size instead of just (i - 1) % size because idk how C modulo works with negatives
            const auto nextFrameIdx = (input.simFrameIndex + data.perFrameData.size() - 1) % data.perFrameData.size();
            const auto& previousFrameData = data.perFrameData[nextFrameIdx];
            waitSemaphores.push_back(*previousFrameData.renderFinishedNextFrameCanCompute);
            waitStages.push_back(vk::PipelineStageFlagBits::eComputeShader);
        }

        submitInfo.waitSemaphoreCount = waitSemaphores.size();
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();

        vk::CommandBuffer cmdBuffers[] = {
            computeCmdBuffer
        };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = cmdBuffers;

        vk::Semaphore signalSemaphores[] = {*simFrameData.computeFinishedCanRender, *simFrameData.computeFinishedCanSim};
        submitInfo.signalSemaphoreCount = 2;
        submitInfo.pSignalSemaphores = signalSemaphores;

        global.context.computeQueue.submit({submitInfo}, nullptr);
    }

    // Send the graphics work
    {
        vk::SubmitInfo submitInfo{};
        vk::Semaphore waitSemaphores[] = {*simFrameData.imageAcquiredCanRender, *simFrameData.computeFinishedCanRender};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTopOfPipe};
        submitInfo.waitSemaphoreCount = 2;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        vk::CommandBuffer cmdBuffers[] = {
            graphicsCmdBuffer
        };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = cmdBuffers;

        vk::Semaphore signalSemaphores[] = {*simFrameData.renderFinishedCanPresent, *simFrameData.renderFinishedNextFrameCanCompute};
        submitInfo.signalSemaphoreCount = 2;
        submitInfo.pSignalSemaphores = signalSemaphores;

        // We are now using these command buffers to execute - don't record over them
        global.device.resetFences({*simFrameData.frameCmdBuffersInUse});
        // Tell the graphics queue to re-open the fence once it's done with this submission
        global.context.graphicsQueue.submit({submitInfo}, *simFrameData.frameCmdBuffersInUse);
    }

    // Send the Present work
    {
        vk::PresentInfoKHR presentInfo{};
        vk::Semaphore presentWaitSemaphores[] = {*simFrameData.renderFinishedCanPresent};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = presentWaitSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &(*global.swapchain);
        presentInfo.pImageIndices = &swapchainImageIdx;
        presentInfo.pResults = nullptr;
        global.context.presentQueue.presentKHR(presentInfo);
    }

    return SystemWorkerOut{
        .wantsQuit = wantsQuit,
        .wantsRunSim = wantsRunSim
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

    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;//oldQueueFamily;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

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

void SystemWorker::fullMemoryBarrier(vk::CommandBuffer cmdBuffer,
                                     vk::PipelineStageFlags oldStage, vk::PipelineStageFlags newStage,
                                     vk::AccessFlags oldAccess, vk::AccessFlags newAccess) {
    auto memoryBarrier = vk::MemoryBarrier{};
    memoryBarrier.srcAccessMask = oldAccess;
    memoryBarrier.dstAccessMask = newAccess;

    cmdBuffer.pipelineBarrier(
            oldStage, newStage, vk::DependencyFlagBits(0), {memoryBarrier}, {}, {}
    );
}
