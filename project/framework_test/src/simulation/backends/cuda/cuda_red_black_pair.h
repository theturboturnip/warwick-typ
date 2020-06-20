//
// Created by samuel on 18/06/2020.
//

#pragma once

#include <cuda_runtime_api.h>

#include "cuda_memory_wrappers.cuh"

template<typename T>
class CUDAUnifiedRedBlackArraySet {
public:
    explicit CUDAUnifiedRedBlackArraySet(CUDAUnified2DArray<T>&& unsplit);

    void split

    const CUDAUnified2DArray<T> unsplit;
    const CUDAUnified2DArray<T> red, black;
};