//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "I2DAllocator.h"

class Host2DAllocator : public I2DAllocator {
protected:
    std::vector<void*> hostPointers;
    AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) override;
public:
    Host2DAllocator();

    void freeAll() override;
    ~Host2DAllocator() override;
};

