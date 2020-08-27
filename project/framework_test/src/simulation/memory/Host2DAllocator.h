//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "I2DAllocator.h"

class Host2DAllocator : public I2DAllocator {
    std::vector<void*> hostPointers;

public:
    Host2DAllocator();

    AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) override;
    void freeAll() override;
    ~Host2DAllocator() override;
};

