//
// Created by samuel on 26/08/2020.
//

#include "VulkanRenderPass.h"

VulkanRenderPass::VulkanRenderPass(vk::Device device, vk::Format colorAttachmentFormat, VulkanRenderPass::Position position, vk::ImageLayout targetLayout)
    : colorAttachmentFormat(colorAttachmentFormat) {
    const bool isStart = (position == Position::PipelineStart || position == Position::PipelineStartAndEnd);
    const bool isEnd = (position == Position::PipelineEnd || position == Position::PipelineStartAndEnd);

    // Define the render pass
    // The color attachment for the render pass is the texture we're rendering to
    auto colorAttachment = vk::AttachmentDescription();
    colorAttachment.format = colorAttachmentFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1; // No MSAA

    if (isStart)
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear; // Clear the contents before rendering
    else
        colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad; // Load the contents
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore; // Store the contents after rendering

    // We don't care about stencils
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;

    if (isStart)
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined; // We don't care what layout it was in before
    else
        colorAttachment.initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
    if (isEnd)
        colorAttachment.finalLayout = targetLayout; // At the end, it should be in the correct layout for presenting
    else
        colorAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

    std::vector<vk::SubpassDependency> dependencies;
    if (isStart) {
        auto writeAccessDependency = vk::SubpassDependency();
        // Wait on the ColorAttachmentOutput stage in an external subpass (read: the swapchain reading the frame)
        writeAccessDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        writeAccessDependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        writeAccessDependency.srcAccessMask = {};// No access mask - we can't access it?
        // Once that stage is over, the transition to write access can occur and stage 0 can start
        writeAccessDependency.dstSubpass = 0;
        writeAccessDependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        writeAccessDependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

        dependencies.push_back(writeAccessDependency);
    }

    // Define a single subpass for the render pass.
    // This subpass *references* the color attachment from the top render pass and renders to it.
    // When it renders to it, it should be in the ColorAttachmentOptimal layout
    auto colorAttachmentReference = vk::AttachmentReference();
    colorAttachmentReference.attachment = 0;
    colorAttachmentReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

    auto subpass = vk::SubpassDescription();
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentReference;

    auto renderpassCreateInfo = vk::RenderPassCreateInfo();
    renderpassCreateInfo.attachmentCount = 1;
    renderpassCreateInfo.pAttachments = &colorAttachment;
    renderpassCreateInfo.subpassCount = 1;
    renderpassCreateInfo.pSubpasses = &subpass;
    renderpassCreateInfo.dependencyCount = dependencies.size();
    renderpassCreateInfo.pDependencies = dependencies.data();

    renderPass = device.createRenderPassUnique(renderpassCreateInfo);
}
