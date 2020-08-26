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

    AllocatedMemory allocate2D(uint32_t width, uint32_t height, size_t elemSize) override;
    void freeAll() override;
    ~CUDA2DUnifiedAllocator() override;
};

