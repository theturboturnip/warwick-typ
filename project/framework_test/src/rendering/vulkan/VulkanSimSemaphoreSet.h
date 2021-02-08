//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "rendering/vulkan/helpers/VulkanSemaphore.h"

struct VulkanSimSemaphoreSet {
public:
    VulkanSemaphore imageCanBeChanged, simFinished, renderFinishedShouldPresent, renderFinishedShouldSim;

    explicit VulkanSimSemaphoreSet(vk::Device);
};