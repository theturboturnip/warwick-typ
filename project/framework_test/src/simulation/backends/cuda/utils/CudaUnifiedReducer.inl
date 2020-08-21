//
// Created by samuel on 18/08/2020.
//

#include "simulation/backends/cuda/kernels/reduction.cuh"

template<size_t BlockSize>
template<typename Preproc, typename Func>
float CudaReducer<BlockSize>::map_reduce(ArrayType& input, Preproc pre, Func func, cudaStream_t stream) {
    FATAL_ERROR_UNLESS(input.raw_length == input_size, "Got input of length %zu, expected %zu", input.raw_length, input_size);

    dim3 blocksize(BlockSize);

    ArrayType* reduction_in = &input;
    ArrayType* reduction_out = &first;
    size_t curr_input_size = input.raw_length;
    bool next_direction = true; // true for (first -> second), false for (second -> first)

    while (curr_input_size > 1) {
        size_t next_output_size = next_reduction_size(curr_input_size);

        dim3 gridsize(next_output_size);

        //printf("Reduce %p [%6zu]-> %p [%6zu]\n", reduction_in, curr_input_size, reduction_out, next_output_size);

        reduce_simple<<<gridsize, blocksize, BlockSize*sizeof(float), stream>>>(reduction_in->as_gpu(), curr_input_size,
                reduction_out->as_gpu(),
                pre, func);
        //cudaStreamSynchronize(stream);

        if (next_direction) {
            reduction_in = &first;
            reduction_out = &second;
        } else {
            reduction_in = &second;
            reduction_out = &first;
        }
        next_direction = !next_direction;

        curr_input_size = next_output_size;
    }

    cudaStreamSynchronize(stream);
    // curr_input_size = 1 - we're finished! reduction_out has a single float with the result of the reduction
    float result = -1;
    cudaMemcpy(&result, reduction_out->as_gpu(), sizeof(float), cudaMemcpyDefault);
    return result;
}
template<size_t BlockSize>
template<typename Func>
float CudaReducer<BlockSize>::reduce(ArrayType& input, Func func, cudaStream_t stream) {
    // Run a reduction with an identity preprocess
    return get_reduction(input, [](float x) { return x; }, func, stream);
}