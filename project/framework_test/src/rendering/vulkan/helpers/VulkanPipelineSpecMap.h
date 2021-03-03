//
// Created by samuel on 25/02/2021.
//

#ifndef FRAMEWORK_TEST_VULKANPIPELINESPECMAP_H
#define FRAMEWORK_TEST_VULKANPIPELINESPECMAP_H

#include <unordered_map>
#include "VulkanPipeline.h"

template<class TSpecEnum>
class VulkanPipelineSpecMap {
    using Underlying = std::underlying_type_t<TSpecEnum>;
    std::unordered_map<Underlying, VulkanPipeline> pipelines;

public:
    VulkanPipelineSpecMap(
        const std::vector<TSpecEnum>& values,

        vk::Device device, vk::RenderPass renderPass,
        Size<uint32_t> viewportSize,
        const VertexShader& vertex, const FragmentShader& fragment,
        VulkanVertexInformation::Kind vertexInfoKind,
        const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts = {},
        size_t pushConstantSize=0
    ) {
        auto specMapEntry = vk::SpecializationMapEntry(
            0, // Spec Constant #0
            0, // Offset
            sizeof(TSpecEnum) // Size
        );
        auto specInfo = vk::SpecializationInfo(1, &specMapEntry, sizeof(TSpecEnum), nullptr);
        for (TSpecEnum value : values) {
            specInfo.pData = &value;
            pipelines.insert(
                {static_cast<Underlying>(value), VulkanPipeline(device, renderPass, viewportSize, vertex, fragment, vertexInfoKind, descriptorSetLayouts, pushConstantSize, specInfo)}
            );
        }
    }
    VulkanPipelineSpecMap(
            const std::vector<TSpecEnum>& values,

            vk::Device device,
            const ComputeShader& compute,
            const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts = {},
            size_t pushConstantSize=0
    ) {
        auto specMapEntry = vk::SpecializationMapEntry(
                0, // Spec Constant #0
                0, // Offset
                sizeof(TSpecEnum) // Size
        );
        auto specInfo = vk::SpecializationInfo(1, &specMapEntry, sizeof(TSpecEnum), nullptr);
        for (TSpecEnum value : values) {
            specInfo.pData = &value;
            pipelines.insert(
                    {static_cast<Underlying>(value), VulkanPipeline(device, compute, descriptorSetLayouts, pushConstantSize, specInfo)}
            );
        }
    }

    const VulkanPipeline& operator[] (TSpecEnum index) {
        return pipelines.at(static_cast<Underlying>(index));
    }
};


#endif //FRAMEWORK_TEST_VULKANPIPELINESPECMAP_H
