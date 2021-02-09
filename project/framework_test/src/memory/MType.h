//
// Created by samuel on 08/02/2021.
//

#pragma once

enum class MType {
    Cpu,
#if CUDA_ENABLED
    VulkanCuda,
    Cuda,
#endif
};