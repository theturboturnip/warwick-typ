//
// Created by samuel on 26/08/2020.
//
#include "I2DAllocator.h"
#include <util/fatal_error.h>

void I2DAllocator::requireHostUsable() {
    FATAL_ERROR_IF(!(usage & MemoryUsage::Host), "I2DAllocator of type %s expected to allocate for Host when it cannot\n", typeid(*this).name())
}
void I2DAllocator::requireDeviceUsable() {
    FATAL_ERROR_IF(!(usage & MemoryUsage::Device), "I2DAllocator of type %s expected to allocate for Device when it cannot\n", typeid(*this).name())
}
