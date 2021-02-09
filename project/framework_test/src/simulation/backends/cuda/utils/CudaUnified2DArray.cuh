//
// Created by samuel on 12/08/2020.
//

#pragma once

#include <cuda_runtime_api.h>

#include <vector>

#include "memory/Sim2DArray.h"

#include "util/fatal_error.h"
#include "util/check_cuda_error.cuh"

template<typename T, bool UnifiedMemory>
using CudaUnified2DArray = Sim2DArray<T, (UnifiedMemory ? MType::Cuda : MType::VulkanCuda)>;
