//
// Created by samuel on 26/08/2020.
//

#include "Host2DAllocator.h"

#include "util/fatal_error.h"

Host2DAllocator::Host2DAllocator()
    : I2DAllocator(MemoryUsage::Host),
      hostPointers()
{}
AllocatedMemory Host2DAllocator::allocate2D(uint32_t width, uint32_t height, size_t elemSize) {
    void* pointer = malloc(width * height * elemSize);
    FATAL_ERROR_IF(!pointer, "Failed to allocate memory\n");
    return AllocatedMemory{
            .pointer = pointer,
            .totalSize = width * height,

            .width = width,
            .height = height,
            .columnStride = height,
    };
}
void Host2DAllocator::freeAll() {
    for (void* pointer : hostPointers) {
        free(pointer);
    }
    hostPointers.clear();
}
Host2DAllocator::~Host2DAllocator() {
    if (!hostPointers.empty())
        freeAll();
}
