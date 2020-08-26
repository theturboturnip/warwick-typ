//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "I2DAllocator.h"

class Host2DAllocator : public I2DAllocator {
    std::vector<void*> hostPointers;

public:
    Host2DAllocator();

    AllocatedMemory allocate2D(uint32_t width, uint32_t height, size_t elemSize) override;
    void freeAll() override;
    ~Host2DAllocator() override;
};

