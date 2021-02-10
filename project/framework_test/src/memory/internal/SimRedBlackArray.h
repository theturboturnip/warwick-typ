//
// Created by samuel on 08/02/2021.
//

#pragma once

#include "Sim2DArray.h"
#include "util/Size.h"
#include "FrameAllocator_fwd.h"

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

template<class T, MType MemType_Template>
class SimRedBlackArray<T, MemType_Template, RedBlackStorage::RedBlackOnly> {
public:
    using ArrayType = Sim2DArray<T, MemType_Template>;
    constexpr static MType MemType = MemType_Template;

    SimRedBlackArray(FrameAllocator<MemType_Template>& alloc, Size<uint32_t> paddedFullSize);

    static size_t sizeBytesOf(Size<uint32_t> paddedFullSize) {
        // Half are red, half are black => total amount of elements is paddedFullSize.area()
        return paddedFullSize.area() * sizeof(T);
    }

    Sim2DArrayStats splitStats;

    ArrayType red;
    ArrayType black;

    ArrayType& get(RedBlack toGet) {
        if (toGet == RedBlack::Red)
            return red;
        else
            return black;
    }

    ArrayType& get_other(RedBlack toGet) {
        if (toGet == RedBlack::Black)
            return red;
        else
            return black;
    }

    // TODO - memcpy_in?

// TODO - enable_if CUDA
    void dispatch_gpu_prefetch(int deviceId, cudaStream_t stream) {
        static_assert(MemType == MType::Cuda, "cudaMemPrefetchAsync only works on Unified Memory");
        red.dispatch_gpu_prefetch(deviceId, stream);
        black.dispatch_gpu_prefetch(deviceId, stream);
    }

    void zero_out() {
        red.zero_out();
        black.zero_out();
    }

protected:
    SimRedBlackArray(ArrayType&& red, ArrayType&& black)
        : splitStats(red.stats),
        red(std::move(red)),
        black(std::move(black)) {}

    friend class FrameAllocator<MemType_Template>;
};

template<class T, MType MemType_Template>
class SimRedBlackArray<T, MemType_Template, RedBlackStorage::WithJoined> : private SimRedBlackArray<T, MemType_Template, RedBlackStorage::RedBlackOnly> {
    using Base = SimRedBlackArray<T, MemType_Template, RedBlackStorage::RedBlackOnly>;
    using BaseArrayType = typename Base::ArrayType;

public:
    BaseArrayType joined;

    SimRedBlackArray(FrameAllocator<MemType_Template>& alloc, Size<uint32_t> paddedFullSize);

    static size_t sizeBytesOf(Size<uint32_t> paddedFullSize) {
        // Half-size red + Half-size black + full-size joined => every pixel is repeated twice

        return paddedFullSize.area() * 2 * sizeof(T);
    }

    using Base::MemType;
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
    void dispatch_gpu_prefetch(int deviceId, cudaStream_t stream) {
        static_assert(MemType == MType::Cuda, "cudaMemPrefetchAsync only works on Unified Memory");
        Base::dispatch_gpu_prefetch(deviceId, stream);
        joined.dispatch_gpu_prefetch(deviceId, stream);
    }

private:
    SimRedBlackArray(BaseArrayType&& joined, BaseArrayType&& red, BaseArrayType&& black)
            : Base(std::move(red), std::move(black)),
              joined(std::move(joined)) {}

    friend class FrameAllocator<MemType>;
};