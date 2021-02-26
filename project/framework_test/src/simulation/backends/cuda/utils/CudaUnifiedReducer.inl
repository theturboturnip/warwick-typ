//
// Created by samuel on 18/08/2020.
//

#include "simulation/backends/cuda/kernels/reduction.cuh"

template<size_t BlockSize>
template<MType MemType, typename Preproc, typename Func>
float CudaReducer<BlockSize>::map_reduce(Sim2DArray<float, MemType>& input, Preproc pre, Func func, cudaStream_t stream) {
    FATAL_ERROR_UNLESS(input.stats.raw_length == input_size, "Got input of length %zu, expected %u", input.stats.raw_length, input_size);

    dim3 blocksize(BlockSize);

    float* reduction_in = input.as_cuda();
    float* reduction_out = first.as_cuda();
    size_t curr_input_size = input.stats.raw_length;
    bool next_direction = true; // true for (first -> second), false for (second -> first)

    while (curr_input_size > 1) {
        size_t next_output_size = next_reduction_size(curr_input_size);

        dim3 gridsize(next_output_size); // TODO - this should be divided by BlockSize?????

        //printf("Reduce %p [%6zu]-> %p [%6zu]\n", reduction_in, curr_input_size, reduction_out, next_output_size);

        reduce_simple<<<gridsize, blocksize, BlockSize*sizeof(float), stream>>>(reduction_in, curr_input_size,
                reduction_out,
                pre, func);
        CHECK_KERNEL_ERROR();
        //cudaStreamSynchronize(stream);

        if (next_direction) {
            reduction_in = first.as_cuda();
            reduction_out = second.as_cuda();
        } else {
            reduction_in = second.as_cuda();
            reduction_out = first.as_cuda();
        }
        next_direction = !next_direction;

        curr_input_size = next_output_size;
    }

    CHECKED_CUDA(cudaStreamSynchronize(stream));
    // TODO - reduction_out won't have the max value, reduction_in will? have they swapped around at this point?
    // curr_input_size = 1 - we're finished! reduction_out has a single float with the result of the reduction
    float result = -1;
    CHECKED_CUDA(cudaMemcpy(&result, reduction_out, sizeof(float), cudaMemcpyDefault));
    return result;
}
template<size_t BlockSize>
template<MType MemType, typename Func>
float CudaReducer<BlockSize>::reduce(Sim2DArray<float, MemType>& input, Func func, cudaStream_t stream) {
    // Run a reduction with an identity preprocess
    return get_reduction(input, [](float x) { return x; }, func, stream);
}