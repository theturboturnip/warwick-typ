//
// Created by samuel on 18/08/2020.
//

#pragma once

#include "CudaUnified2DArray.cuh"

// PSEUDOCODE DISPATCH IDEA
// Preallocate 2 buffers
// first reduction makes ceil(input / blocksize) results => first buffer size has to be ceil(input size / blocksize)
// second reduction is ceil(ceil(input / blocksize) / blocksize)
//  note that the ceil is done twice there
template<size_t BlockSize>
class CudaReducer {
    using ArrayType = CudaUnified2DArray<float>;

    const size_t input_size;
    ArrayType first;
    ArrayType second;
public:
    explicit CudaReducer(size_t input_size)
        : input_size(input_size),
          first(next_reduction_size(input_size), 1),
          second(next_reduction_size(first.raw_length), 1){}

    template<typename Preproc, typename Func>
    float map_reduce(ArrayType& input, Preproc pre, Func func, cudaStream_t stream);
    template<typename Func>
    float reduce(ArrayType& input, Func func, cudaStream_t stream);

    static size_t next_reduction_size(size_t input_size) {
        return (input_size + BlockSize - 1) / BlockSize;
    }
};

#if __CUDACC__
#include "CudaUnifiedReducer.inl"
#endif