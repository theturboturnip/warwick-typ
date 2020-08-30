//
// Created by samuel on 18/08/2020.
//

#include "simulation/backends/cuda/kernels/reduction.cuh"

template<bool UnifiedMemoryForBacking, size_t BlockSize>
template<bool UnifiedMemory, typename Preproc, typename Func>
float CudaReducer<UnifiedMemoryForBacking, BlockSize>::map_reduce(CudaUnified2DArray<float, UnifiedMemory>& input, Preproc pre, Func func, cudaStream_t stream) {
    FATAL_ERROR_UNLESS(input.raw_length == input_size, "Got input of length %zu, expected %u", input.raw_length, input_size);

    dim3 blocksize(BlockSize);

    float* reduction_in = input.as_gpu();
    float* reduction_out = first.as_gpu();
    size_t curr_input_size = input.raw_length;
    bool next_direction = true; // true for (first -> second), false for (second -> first)

    while (curr_input_size > 1) {
        size_t next_output_size = next_reduction_size(curr_input_size);

        dim3 gridsize(next_output_size);

        //printf("Reduce %p [%6zu]-> %p [%6zu]\n", reduction_in, curr_input_size, reduction_out, next_output_size);

        reduce_simple<<<gridsize, blocksize, BlockSize*sizeof(float), stream>>>(reduction_in, curr_input_size,
                reduction_out,
                pre, func);
        CHECK_KERNEL_ERROR();
        //cudaStreamSynchronize(stream);

        if (next_direction) {
            reduction_in = first.as_gpu();
            reduction_out = second.as_gpu();
        } else {
            reduction_in = second.as_gpu();
            reduction_out = first.as_gpu();
        }
        next_direction = !next_direction;

        curr_input_size = next_output_size;
    }

    CHECKED_CUDA(cudaStreamSynchronize(stream));
    // curr_input_size = 1 - we're finished! reduction_out has a single float with the result of the reduction
    float result = -1;
    CHECKED_CUDA(cudaMemcpy(&result, reduction_out, sizeof(float), cudaMemcpyDefault));
    return result;
}
template<bool UnifiedMemoryForBacking, size_t BlockSize>
template<bool UnifiedMemory, typename Func>
float CudaReducer<UnifiedMemoryForBacking, BlockSize>::reduce(CudaUnified2DArray<float, UnifiedMemory>& input, Func func, cudaStream_t stream) {
    // Run a reduction with an identity preprocess
    return get_reduction(input, [](float x) { return x; }, func, stream);
}