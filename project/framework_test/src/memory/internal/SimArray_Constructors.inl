//
// Created by samuel on 10/02/2021.
//

#pragma once

#include "memory/FrameAllocator.h"

template<class T>
Sim2DArray<T, MType::Cpu>::Sim2DArray(FrameAllocator<MType::Cpu>& alloc, Size<uint32_t> size)
        : Sim2DArray(alloc.allocate2D<T>(size)) {}

#if CUDA_ENABLED
template<class T>
Sim2DArray<T, MType::Cuda>::Sim2DArray(FrameAllocator<MType::Cuda>& alloc, Size<uint32_t> size)
        : Sim2DArray(alloc.allocate2D<T>(size)) {}

template<class T>
Sim2DArray<T, MType::VulkanCuda>::Sim2DArray(FrameAllocator<MType::VulkanCuda>& alloc, Size<uint32_t> size)
        : Sim2DArray(alloc.allocate2D<T>(size)) {}
#endif

template<class T, MType MemType_Template>
SimRedBlackArray<T, MemType_Template, RedBlackStorage::RedBlackOnly>::SimRedBlackArray(FrameAllocator<MemType_Template> &alloc,
                                                                                       Size<uint32_t> paddedFullSize)
        : SimRedBlackArray(alloc.template allocateRedBlack<T, RedBlackStorage::RedBlackOnly>(paddedFullSize)) {}

template<class T, MType MemType_Template>
SimRedBlackArray<T, MemType_Template, RedBlackStorage::WithJoined>::SimRedBlackArray(FrameAllocator<MemType_Template> &alloc,
                                                                                     Size<uint32_t> paddedFullSize)
        : SimRedBlackArray(alloc.template allocateRedBlack<T, RedBlackStorage::WithJoined>(paddedFullSize)) {}