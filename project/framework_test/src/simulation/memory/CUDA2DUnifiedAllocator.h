//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "I2DAllocator.h"

#include <vector>

class CUDA2DUnifiedAllocator : public I2DAllocator {
    std::vector<void*> cudaPointers;

public:
    CUDA2DUnifiedAllocator();

    AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) override;
    void freeAll() override;
    ~CUDA2DUnifiedAllocator() override;
};

