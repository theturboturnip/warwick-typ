//
// Created by samuel on 15/08/2020.
//

#pragma once

#include "CudaUnified2DArray.cuh"

enum class RedBlack : bool {
    Red = true,
    Black = false
};

enum class RedBlackStorage {
    RedBlackOnly,
    WithJoined
};

template<typename T, bool UnifiedMemory, RedBlackStorage Storage=RedBlackStorage::WithJoined>
class CudaUnifiedRedBlackArray;

template<typename T, bool UnifiedMemory>
class CudaUnifiedRedBlackArray<T, UnifiedMemory, RedBlackStorage::RedBlackOnly> {
public:
    using ArrayType = CudaUnified2DArray<T, UnifiedMemory>;

    Size<uint32_t> split_size;
    ArrayType red;
    ArrayType black;

    explicit CudaUnifiedRedBlackArray(I2DAllocator* alloc, Size<uint32_t> full_size)
        : split_size(full_size.x, full_size.y / 2),
          red(alloc, split_size),
          black(alloc, split_size)
    {}

    template<RedBlack ToGet>
    ArrayType& get() {
        if (ToGet == RedBlack::Red)
            return red;
        else
            return black;
    }

    template<RedBlack ToGet>
    ArrayType& get_other() {
        if (ToGet == RedBlack::Black)
            return red;
        else
            return black;
    }

    void dispatch_gpu_prefetch(int deviceId, cudaStream_t stream) {
        static_assert(UnifiedMemory, "cudaMemPrefetchAsync only works on Unified Memory");
        red.dispatch_gpu_prefetch(deviceId, stream);
        black.dispatch_gpu_prefetch(deviceId, stream);
    }

    void zero_out() {
        red.zero_out();
        black.zero_out();
    }
};

template<typename T, bool UnifiedMemory>
class CudaUnifiedRedBlackArray<T, UnifiedMemory, RedBlackStorage::WithJoined> : private CudaUnifiedRedBlackArray<T, UnifiedMemory, RedBlackStorage::RedBlackOnly> {
    using Base = CudaUnifiedRedBlackArray<T, UnifiedMemory, RedBlackStorage::RedBlackOnly>;

public:
    typename Base::ArrayType joined;
    using Base::red;
    using Base::black;
    using Base::get;
    using Base::get_other;

    explicit CudaUnifiedRedBlackArray(I2DAllocator* alloc, Size<uint32_t> full_size)
            : Base(alloc, full_size),
              joined(alloc, full_size)
    {}
    explicit CudaUnifiedRedBlackArray(I2DAllocator* alloc, AllocatedMemory<T> joined)
            : Base(alloc, {joined.width, joined.height}),
              joined(alloc, joined)
    {}

    // Virtual functions are not required here because this is never downcasted

    void zero_out() {
        Base::zero_out();
        joined.zero_out();
    }

    void dispatch_gpu_prefetch(int deviceId, cudaStream_t stream) {
        Base::dispatch_gpu_prefetch(deviceId, stream);
        joined.dispatch_gpu_prefetch(deviceId, stream);
    }
};