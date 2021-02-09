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
    const uint32_t input_size;
    CudaUnified2DArray<float, true> first;
    CudaUnified2DArray<float, true> second;
public:
    explicit CudaReducer(FrameAllocator<MType::Cuda>& alloc, uint32_t input_size)
        : input_size(input_size),
          first(alloc.allocate2D(first, {next_reduction_size(input_size), 1})),
          second(alloc.allocate2D(second, {next_reduction_size(first.stats.raw_length), 1})){}

    template<bool UnifiedMemory, typename Preproc, typename Func>
    float map_reduce(CudaUnified2DArray<float, UnifiedMemory>& input, Preproc pre, Func func, cudaStream_t stream);
    template<bool UnifiedMemory, typename Func>
    float reduce(CudaUnified2DArray<float, UnifiedMemory>& input, Func func, cudaStream_t stream);

    static uint32_t next_reduction_size(uint32_t input_size) {
        return (input_size + BlockSize - 1) / BlockSize;
    }
};

#if __CUDACC__
#include "CudaUnifiedReducer.inl"
#endif