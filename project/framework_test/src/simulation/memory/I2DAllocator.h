//
// Created by samuel on 26/08/2020.
//

#pragma once

#include <cstddef>
#include <cstdint>

struct AllocatedMemory {
    void* pointer = nullptr;
    size_t totalSize = -1;

    uint32_t width = -1;
    uint32_t height = -1;
    uint32_t columnStride = -1;
};

class I2DAllocator {
protected:
    I2DAllocator(uint32_t usage) : usage(usage) {}
public:
    const uint32_t usage;
    enum MemoryUsage {
        Host   = 0b01,
        Device = 0b10
    };

    // TODO - add "pitched" argument here
    virtual AllocatedMemory allocate2D(uint32_t width, uint32_t height, size_t elemSize) = 0;
    virtual void freeAll() = 0;
    virtual ~I2DAllocator() = default;
};