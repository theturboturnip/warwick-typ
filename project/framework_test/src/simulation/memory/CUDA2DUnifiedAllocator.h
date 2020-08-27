//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "I2DAllocator.h"

#include <vector>

class CUDA2DUnifiedAllocator : public I2DAllocator {
protected:
    std::vector<void*> cudaPointers;
    AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) override;
public:
    CUDA2DUnifiedAllocator();

    void freeAll() override;
    ~CUDA2DUnifiedAllocator() override;
};

