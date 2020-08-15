//
// Created by samuel on 15/08/2020.
//

#pragma once

#include "CudaUnified2DArray.cuh"

enum class RedBlack {
    Red,
    Black
};

enum class RedBlackStorage {
    RedBlackOnly,
    WithJoined
};

template<typename T, RedBlackStorage Storage=RedBlackStorage::WithJoined, CudaMemoryType MemoryType=CudaMemoryType::CudaManaged>
class CudaUnifiedRedBlackArray;

template<typename T, CudaMemoryType MemoryType>
class CudaUnifiedRedBlackArray<T, RedBlackStorage::RedBlackOnly, MemoryType> {
protected:
    using ArrayType = CudaUnified2DArray<T, MemoryType>;

    Size<size_t> split_size;
    ArrayType red;
    ArrayType black;

public:
    explicit CudaUnifiedRedBlackArray(Size<size_t> full_size)
        : split_size(full_size.x, full_size.y / 2),
          red(split_size),
          black(split_size)
    {}

    template<RedBlack ToGet>
    ArrayType& get() {
        if (ToGet == RedBlack::Red)
            return red;
        else
            return black;
    }

    void zero_out() {
        red.zero_out();
        black.zero_out();
    }
};

template<typename T, CudaMemoryType MemoryType>
class CudaUnifiedRedBlackArray<T, RedBlackStorage::WithJoined, MemoryType> : public CudaUnifiedRedBlackArray<T, RedBlackStorage::RedBlackOnly, MemoryType> {
    using Base = CudaUnifiedRedBlackArray<T, RedBlackStorage::RedBlackOnly, MemoryType>;

    typename Base::ArrayType joined;

public:
    explicit CudaUnifiedRedBlackArray(Size<size_t> full_size)
            : Base(full_size),
              joined(full_size)
    {}

    typename Base::ArrayType& get_joined() {
        return joined;
    }

    // This hides Base::zero_out, but this class is non-virtual so that's OK
    void zero_out() {
        Base::zero_out();
        joined.zero_out();
    }
};