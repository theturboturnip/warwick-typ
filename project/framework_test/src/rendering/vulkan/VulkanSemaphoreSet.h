//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "VulkanSemaphore.h"

struct VulkanSemaphoreSet {
public:
    VulkanSemaphore imageCanBeChanged, simFinished, renderFinishedShouldPresent, renderFinishedShouldSim;

    VulkanSemaphoreSet(vk::Device);
};