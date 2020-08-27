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
AllocatedMemory<void> CUDA2DUnifiedAllocator::allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) {
    const size_t sizeBytes = size.x * size.y * elemSize;
    void* pointer = nullptr;
    auto error = cudaMallocManaged(&pointer, sizeBytes * elemSize);
    if (error != cudaSuccess || !pointer)
        FATAL_ERROR("CUDA Alloc Error: %s\n", cudaGetErrorString(error));

    if (initialData) {
        cudaMemcpy(pointer, initialData, sizeBytes, cudaMemcpyDefault);
    } else {
        cudaMemset(pointer, 0, sizeBytes);
    }
    // TODO - pitch allocation
    //  pitched arrays MUST have zeroes in the padding for reductions to work - note this
    return AllocatedMemory<void>{
            .pointer = pointer,
            .totalSize = size.x * size.y,

            .width = size.x,
            .height = size.y,
            .columnStride = size.y,
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
