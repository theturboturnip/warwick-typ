//
// Created by samuel on 15/08/2020.
//

#pragma once

#include "memory/FrameAllocator.h"

template<typename T, bool UnifiedMemory, RedBlackStorage Storage=RedBlackStorage::WithJoined>
using CudaUnifiedRedBlackArray = SimRedBlackArray<T, (UnifiedMemory ? MType::Cuda : MType::VulkanCuda), Storage>;

