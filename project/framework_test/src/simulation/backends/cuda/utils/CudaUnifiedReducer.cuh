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
    Sim2DArray<float, MType::Cuda> first;
    Sim2DArray<float, MType::Cuda> second;
public:
    explicit CudaReducer(FrameAllocator<MType::Cuda>& alloc, uint32_t input_size)
        : input_size(input_size),
          first(alloc, {next_reduction_size(input_size), 1}),
          second(alloc, {next_reduction_size(first.stats.raw_length), 1}){}

    template<MType MemType, typename Preproc, typename Func>
    float map_reduce(Sim2DArray<float, MemType>& input, Preproc pre, Func func, cudaStream_t stream);
    template<MType MemType, typename Func>
    float reduce(Sim2DArray<float, MemType>& input, Func func, cudaStream_t stream);

    static uint32_t next_reduction_size(uint32_t input_size) {
        return (input_size + BlockSize - 1) / BlockSize;
    }
};

#if __CUDACC__
#include "CudaUnifiedReducer.inl"
#endif