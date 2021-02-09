//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "Sim2DArray.h"
#include "util/Size.h"

enum class RedBlack {
    Red,
    Black
};

enum class RedBlackStorage {
    RedBlackOnly,
    WithJoined
};

template<class T, MType MemType, RedBlackStorage Storage=RedBlackStorage::WithJoined>
class SimRedBlackArray;

template<class T, MType MemType>
class SimRedBlackArray<T, MemType, RedBlackStorage::RedBlackOnly> {
public:
    using ArrayType = SimRedBlackArray<T, MemType>;

    Sim2DArrayStats splitStats;

    ArrayType red;
    ArrayType black;

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

    // TODO - memcpy_in?

// TODO - enable_if CUDA
//    void dispatch_gpu_prefetch(int deviceId, cudaStream_t stream) {
//        static_assert(UnifiedMemory, "cudaMemPrefetchAsync only works on Unified Memory");
//        red.dispatch_gpu_prefetch(deviceId, stream);
//        black.dispatch_gpu_prefetch(deviceId, stream);
//    }

    void zero_out() {
        red.zero_out();
        black.zero_out();
    }

private:
    SimRedBlackArray(ArrayType&& red, ArrayType&& black)
        : splitStats(red.stats),
        red(red),
        black(black) {}
    SimRedBlackArray(SimRedBlackArray&&) = default;
    SimRedBlackArray(const SimRedBlackArray&) = delete;

    friend class Allocator<MemType>;
};

template<class T, MType MemType>
class SimRedBlackArray<T, MemType, RedBlackStorage::WithJoined> : private SimRedBlackArray<T, MemType, RedBlackStorage::RedBlackOnly> {
    using Base = SimRedBlackArray<T, MemType, RedBlackStorage::RedBlackOnly>;
    using BaseArrayType = typename Base::ArrayType;

public:
    BaseArrayType joined;

    using Base::red;
    using Base::black;
    using Base::get;
    using Base::get_other;

    // Virtual functions are not required here because this is never downcasted

    void zero_out() {
        Base::zero_out();
        joined.zero_out();
    }

    // TODO - memcpy_in?

// TODO - enable_if CUDA
//    void dispatch_gpu_prefetch(int deviceId, cudaStream_t stream) {
//        Base::dispatch_gpu_prefetch(deviceId, stream);
//        joined.dispatch_gpu_prefetch(deviceId, stream);
//    }

private:
    SimRedBlackArray(BaseArrayType&& joined, BaseArrayType&& red, BaseArrayType&& black)
            : Base(red, black),
              joined(joined) {}
    SimRedBlackArray(SimRedBlackArray&&) = default;
    SimRedBlackArray(const SimRedBlackArray&) = delete;

    friend class Allocator<MemType>;
};