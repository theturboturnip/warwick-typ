//
// Created by samuel on 24/08/2020.
//

#include "VulkanPipeline.h"


VulkanPipeline::VulkanPipeline(
        vk::Device device, vk::RenderPass renderPass,
        Size<uint32_t> viewportSize,
        const VertexShader &vertex, const FragmentShader &fragment,
        const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
        size_t pushConstantSize,
        vk::SpecializationInfo specInfo) {
    {
        pushConstantRange = vk::PushConstantRange();
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        pushConstantRange.size = pushConstantSize;
        pushConstantRange.offset = 0;
    }

    {
        auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();
        pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();// Descriptor sets
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

        pipelineLayoutInfo.pushConstantRangeCount = (pushConstantSize ? 1 : 0);// Push constants
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        layout = device.createPipelineLayoutUnique(pipelineLayoutInfo);
    }

    auto vertexInput = vk::PipelineVertexInputStateCreateInfo();
    // Assume vertices are hardcoded - TODO make this VertexInputStateCreateInfo a static value with a name like "hardcodedVertexInput"
    vertexInput.vertexBindingDescriptionCount = 0;
    vertexInput.pVertexBindingDescriptions = nullptr;
    vertexInput.vertexAttributeDescriptionCount = 0;
    vertexInput.pVertexAttributeDescriptions = nullptr;

    auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo();
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo();
    rasterizer.depthClampEnable = VK_FALSE; // Discard fragments that are outside [0,1] depth
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // Don't just stop rasterizing
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f; // Pixel width of lines, if line rendering used.
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    auto viewport = vk::Viewport();
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) viewportSize.x;
    viewport.height = (float) viewportSize.y;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    auto scissor = vk::Rect2D();
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = vk::Extent2D{ viewportSize.x, viewportSize.y };
    auto viewportState = vk::PipelineViewportStateCreateInfo();
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    auto multisampling = vk::PipelineMultisampleStateCreateInfo();
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState();
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne; // Optional
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero; // Optional
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero; // Optional
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd; // Optional

    auto colorBlending = vk::PipelineColorBlendStateCreateInfo();
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    auto shaderStages = std::vector<vk::PipelineShaderStageCreateInfo>({
            vertex.getShaderStage(&specInfo),
            fragment.getShaderStage(&specInfo),
    });

    // TODO - make these derive from a common base pipeline? Not worth it right now
    auto pipelineInfo = vk::GraphicsPipelineCreateInfo();
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr; // Optional
    pipelineInfo.layout = *layout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    pipeline = device.createGraphicsPipelineUnique(nullptr, {pipelineInfo});
}

VulkanPipeline::VulkanPipeline(
        vk::Device device,
        const ComputeShader &compute,
        const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
        size_t pushConstantSize,
        vk::SpecializationInfo specInfo) {
    {
        pushConstantRange = vk::PushConstantRange();
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pushConstantRange.size = pushConstantSize;
        pushConstantRange.offset = 0;
    }

    {
        auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();
        pipelineLayoutInfo.setLayoutCount = descriptorSetLayouts.size();// Descriptor sets
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

        pipelineLayoutInfo.pushConstantRangeCount = (pushConstantSize ? 1 : 0);// Push constants
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        layout = device.createPipelineLayoutUnique(pipelineLayoutInfo);
    }

    auto pipelineInfo = vk::ComputePipelineCreateInfo();
    pipelineInfo.layout = *layout;
    pipelineInfo.stage = compute.getShaderStage(&specInfo);
    pipelineInfo.basePipelineHandle = nullptr; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    pipeline = device.createComputePipelineUnique(nullptr, {pipelineInfo});
}
