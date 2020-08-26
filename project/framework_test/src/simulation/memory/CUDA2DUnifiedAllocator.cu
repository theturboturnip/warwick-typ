//
// Created by samuel on 26/08/2020.
//

#include "CUDA2DUnifiedAllocator.h"

#include "util/fatal_error.h"

CUDA2DUnifiedAllocator::CUDA2DUnifiedAllocator()
    : I2DAllocator(MemoryUsage::Host | MemoryUsage::Device),
      cudaPointers()
{
    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);
    FATAL_ERROR_IF(deviceCount <= 0, "No CUDA Device present to allocate on!\n");
}
AllocatedMemory CUDA2DUnifiedAllocator::allocate2D(uint32_t width, uint32_t height, size_t elemSize) {
    void* pointer = nullptr;
    auto error = cudaMallocManaged(&pointer, width * height * elemSize);
    if (error != cudaSuccess)
        FATAL_ERROR("CUDA Alloc Error: %s\n", cudaGetErrorString(error));
    DASSERT(pointer);
    // TODO - pitch allocation
    //  pitched arrays MUST have zeroes in the padding for reductions to work - note this
    return AllocatedMemory{
            .pointer = pointer,
            .totalSize = width * height,

            .width = width,
            .height = height,
            .columnStride = height,
    };
}
void CUDA2DUnifiedAllocator::freeAll() {
    for (void* pointer : cudaPointers) {
        cudaFree(pointer);
    }
    cudaPointers.clear();
}
CUDA2DUnifiedAllocator::~CUDA2DUnifiedAllocator() {
    if (!cudaPointers.empty())
        freeAll();
}
