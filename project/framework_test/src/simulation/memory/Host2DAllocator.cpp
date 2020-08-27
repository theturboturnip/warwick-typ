//
// Created by samuel on 26/08/2020.
//

#include "Host2DAllocator.h"

#include "util/fatal_error.h"

Host2DAllocator::Host2DAllocator()
    : I2DAllocator(MemoryUsage::Host),
      hostPointers()
{}
AllocatedMemory<void> Host2DAllocator::allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) {
    const size_t sizeBytes = size.x * size.y * elemSize;
    void* pointer = malloc(sizeBytes);
    FATAL_ERROR_IF(!pointer, "Failed to allocate memory\n");
    if (initialData) {
        memcpy(pointer, initialData, sizeBytes);
    } else {
        memset(pointer, 0, sizeBytes);
    }
    return AllocatedMemory<void>{
            .pointer = pointer,
            .totalSize = size.x * size.y,

            .width = size.x,
            .height = size.y,
            .columnStride = size.y,
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
