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

template<typename T, RedBlackStorage Storage=RedBlackStorage::WithJoined, CudaMemoryType MemoryType=CudaMemoryType::CudaManaged>
class CudaUnifiedRedBlackArray;

template<typename T, CudaMemoryType MemoryType>
class CudaUnifiedRedBlackArray<T, RedBlackStorage::RedBlackOnly, MemoryType> {
public:
    using ArrayType = CudaUnified2DArray<T, MemoryType>;

    Size<size_t> split_size;
    ArrayType red;
    ArrayType black;

    explicit CudaUnifiedRedBlackArray(Size<size_t> full_size)
        : split_size(full_size.x, full_size.y / 2),
          red(split_size),
          black(split_size)
    {}
    virtual ~CudaUnifiedRedBlackArray() = default;

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

    virtual void zero_out() {
        red.zero_out();
        black.zero_out();
    }
};

template<typename T, CudaMemoryType MemoryType>
class CudaUnifiedRedBlackArray<T, RedBlackStorage::WithJoined, MemoryType> : public CudaUnifiedRedBlackArray<T, RedBlackStorage::RedBlackOnly, MemoryType> {
    using Base = CudaUnifiedRedBlackArray<T, RedBlackStorage::RedBlackOnly, MemoryType>;

public:
    typename Base::ArrayType joined;

    explicit CudaUnifiedRedBlackArray(Size<size_t> full_size)
            : Base(full_size),
              joined(full_size)
    {}
    ~CudaUnifiedRedBlackArray() override = default;

    void zero_out() override {
        Base::zero_out();
        joined.zero_out();
    }
};