//
// Created by samuel on 26/02/2021.
//

#include "VulkanMinMaxReducer.h"

VulkanMinMaxReducer::DescriptorSets::DescriptorSets(VulkanContext& context,
                                                    VulkanSimPipelineSet &pipelineSet,
                                                    const VulkanBackedBuffer &buffer)
    : rawBuffer(*buffer),
      buffer_comp_ds(pipelineSet.buildBuffer_comp_ds(context, buffer.asDescriptor())),
      buffer_vert_ds(pipelineSet.buildBuffer_vert_ds(context, buffer.asDescriptor())),
      buffer_frag_ds(pipelineSet.buildBuffer_frag_ds(context, buffer.asDescriptor()))
    {}

void VulkanMinMaxReducer::enqueueReductionFromInput(vk::CommandBuffer cmdBuffer, vk::Buffer copyOutputTo) {
    DescriptorSets* reduction_in = &pingPong1_ds;
    DescriptorSets* reduction_out = &pingPong2_ds;
    uint32_t curr_input_count = topLevelCount;

    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineSet.computeMinMaxReduce);
    while (curr_input_count > 1) {
        uint32_t next_output_count = next_reduction_count(curr_input_count);

        // Memory barrier to make sure the last reduction is visible.
        // TODO - this should use a common function?
        // Ensure ShaderWrites from ComputeShader stages are visible to ShaderReads from ComputeShader
        auto memoryBarrier = vk::MemoryBarrier{};
        memoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        memoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits(0), {memoryBarrier}, {}, {}
        );

        // Setup push constants + swap descriptor sets
        auto params = Shaders::MinMaxReduceParams{
            .bufferLength = curr_input_count
        };
        cmdBuffer.pushConstants(
                *pipelineSet.computeMinMaxReduce.layout,
                vk::ShaderStageFlagBits::eCompute,
                0,
                vk::ArrayProxy<const Shaders::MinMaxReduceParams>{params});
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     *pipelineSet.computeMinMaxReduce.layout,
                                    0,
                                    {
                                            *reduction_in->buffer_comp_ds,
                                            *reduction_out->buffer_comp_ds,
                                    },
                                    {});

        //printf("Reduce %p [%6zu]-> %p [%6zu]\n", reduction_in, curr_input_count, reduction_out, next_output_count);
        cmdBuffer.dispatch((curr_input_count + BlockSize - 1) / BlockSize, 1, 1);

//        reduce_simple<<<gridsize, blocksize, BlockSize*sizeof(float), stream>>>(reduction_in, curr_input_count,
//                reduction_out,
//                pre, func);
        
        

        std::swap(reduction_in, reduction_out);

        curr_input_count = next_output_count;
    }

    // Memory barrier so the transfer can move the data to the output
    auto memoryBarrier = vk::MemoryBarrier{};
    memoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    memoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits(0), {memoryBarrier}, {}, {}
    );

    // Just copy the first floatrange.
    // A swap takes place at the end of the loop, so the last output of the reduction is currently in reduction_in
    auto copyRange = vk::BufferCopy(0, 0, sizeof(Shaders::FloatRange));
    cmdBuffer.copyBuffer(reduction_in->rawBuffer, copyOutputTo, {copyRange});
}

