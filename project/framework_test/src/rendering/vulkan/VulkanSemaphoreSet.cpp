//
// Created by samuel on 27/08/2020.
//
#include "VulkanSemaphoreSet.h"

VulkanSemaphoreSet::VulkanSemaphoreSet(vk::Device device)
    : imageCanBeChanged(device),
      simFinished(device),
      renderFinishedShouldPresent(device),
      renderFinishedShouldSim(device)
{}
