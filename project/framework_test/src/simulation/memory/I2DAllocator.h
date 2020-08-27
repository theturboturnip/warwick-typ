//
// Created by samuel on 26/08/2020.
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "util/Size.h"
#include "util/fatal_error.h"

template<typename T>
struct AllocatedMemory {
    T* pointer = nullptr;
    size_t totalSize = 0;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t columnStride = 0;
};

class I2DAllocator {
protected:
    explicit I2DAllocator(uint32_t usage) : usage(usage) {}
public:
    const uint32_t usage;
    enum MemoryUsage {
        Host   = 0b01,
        Device = 0b10
    };

    void requireHostUsable();
    void requireDeviceUsable();

    template<typename T, typename = typename std::enable_if_t<!std::is_same_v<T, void>>>
    AllocatedMemory<T> allocate2D(Size<uint32_t> size, const std::vector<T>* initialData = nullptr) {
        if (initialData) {
            FATAL_ERROR_IF(size.x * size.y != initialData->size(), "Expected size.x*y to be equal to initial data size\n");
        }
        AllocatedMemory<void> alloc = allocate2D_unsafe(size, sizeof(T), initialData ? initialData->data() : nullptr);
        return AllocatedMemory<T> {
                .pointer = reinterpret_cast<T*>(alloc.pointer),
                .totalSize = alloc.totalSize,

                .width = alloc.width,
                .height = alloc.height,
                .columnStride = alloc.columnStride,
        };
    }
    // TODO - add "pitched" argument here
    virtual AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) = 0;
    virtual void freeAll() = 0;
    virtual ~I2DAllocator() = default;
};