//
// Created by samuel on 26/02/2021.
//

#pragma once


#include <rendering/vulkan/helpers/VulkanBackedBuffer.h>
#include <rendering/shaders/global_structures.h>
#include <util/fatal_error.h>
#include "VulkanSimPipelineSet.h"

class VulkanMinMaxReducer {
public:
    struct DescriptorSets {
        vk::Buffer rawBuffer;
        vk::UniqueDescriptorSet buffer_comp_ds;
        // TODO - make uniform-compatible? unnecessary
        vk::UniqueDescriptorSet buffer_vert_ds;
        vk::UniqueDescriptorSet buffer_frag_ds;

        DescriptorSets(VulkanContext& context, VulkanSimPipelineSet& pipelineSet, const VulkanBackedBuffer& buffer);
    };

    constexpr static uint32_t BlockSize = 64; // Has to match compute_minmax_reduce.comp
    static uint32_t next_reduction_count(uint32_t input_count) {
        return (input_count + BlockSize - 1) / BlockSize;
    }

private:
    // This is accessible by the outside, unlike the reducer for Cuda.
    VulkanBackedBuffer pingPong1;
    DescriptorSets pingPong1_ds;
    VulkanBackedBuffer pingPong2;
    DescriptorSets pingPong2_ds;

    VulkanSimPipelineSet& pipelineSet;

    // Number of elements we are built to reduce.
    size_t topLevelCount;

public:

    VulkanMinMaxReducer(VulkanContext& context, VulkanSimPipelineSet& pipelineSet, size_t count, bool shared=false)
        : pingPong1(context,
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    count * sizeof(Shaders::FloatRange), shared),
          pingPong1_ds(context, pipelineSet, pingPong1),
          pingPong2(context,
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                    next_reduction_count(count) * sizeof(Shaders::FloatRange), shared),
          pingPong2_ds(context, pipelineSet, pingPong2),
          pipelineSet(pipelineSet),
          topLevelCount(count)
    {}

    // Get the descriptor sets to output FloatRange values into.
    DescriptorSets& getInputDescriptorSets() {
        return pingPong1_ds;
    }

    // Enqueue a reduction operation onto the command buffer,
    // starting from the input buffer exposed by getInputDescriptorSets.
    // At the end of the reduction, the memory will be transferred into the target buffer.
    void enqueueReductionFromInput(vk::CommandBuffer cmdBuffer, vk::Buffer copyOutputTo);
};
