//
// Created by samuel on 15/08/2020.
//

#pragma once

#include "memory/FrameAllocator.h"

template<typename T, RedBlackStorage Storage=RedBlackStorage::WithJoined>
using CudaUnifiedRedBlackArray = SimRedBlackArray<T, MType::Cuda, Storage>;

