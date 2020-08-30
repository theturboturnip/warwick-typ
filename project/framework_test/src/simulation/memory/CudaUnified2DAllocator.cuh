//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "I2DAllocator.h"

#include <vector>

class CudaUnified2DAllocator : public I2DAllocator {
protected:
    std::vector<void*> cudaPointers;
    AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) override;
public:
    CudaUnified2DAllocator();

    void freeAll() override;
    ~CudaUnified2DAllocator() override;
};

