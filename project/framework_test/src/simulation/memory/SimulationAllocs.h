//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "I2DAllocator.h"

struct SimulationAllocs {
    I2DAllocator* alloc = nullptr; // Can be used to create further allocations that aren't visible to the owner

    Size<uint32_t> matrixSize = {0, 0};

    AllocatedMemory<float> u, v;
    AllocatedMemory<float> p;
    AllocatedMemory<uint32_t> fluidmask;
};