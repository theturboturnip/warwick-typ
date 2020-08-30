//
// Created by samuel on 26/08/2020.
//

#include "CudaUnified2DAllocator.cuh"
#include <util/check_cuda_error.cuh>

#include "util/fatal_error.h"

CudaUnified2DAllocator::CudaUnified2DAllocator()
    : I2DAllocator(MemoryUsage::Host | MemoryUsage::Device),
      cudaPointers()
{
    int deviceCount = -1;
    CHECKED_CUDA(cudaGetDeviceCount(&deviceCount));
    FATAL_ERROR_IF(deviceCount <= 0, "No CUDA Device present to allocate on!\n");
}
AllocatedMemory<void> CudaUnified2DAllocator::allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) {
    const size_t sizeBytes = size.x * size.y * elemSize;
    void* pointer = nullptr;
    CHECKED_CUDA(cudaMallocManaged(&pointer, sizeBytes * elemSize));
    DASSERT(pointer);

    if (initialData) {
        CHECKED_CUDA(cudaMemcpy(pointer, initialData, sizeBytes, cudaMemcpyDefault));
    } else {
        CHECKED_CUDA(cudaMemset(pointer, 0, sizeBytes));
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
void CudaUnified2DAllocator::freeAll() {
    for (void* pointer : cudaPointers) {
        CHECKED_CUDA(cudaFree(pointer));
    }
    cudaPointers.clear();
}
CudaUnified2DAllocator::~CudaUnified2DAllocator() {
    if (!cudaPointers.empty())
        freeAll();
}
