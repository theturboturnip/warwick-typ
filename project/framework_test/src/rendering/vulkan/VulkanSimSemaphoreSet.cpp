//
// Created by samuel on 27/08/2020.
//
#include "VulkanSimSemaphoreSet.h"

VulkanSimSemaphoreSet::VulkanSimSemaphoreSet(vk::Device device)
    : imageCanBeChanged(device),
      simFinished(device),
      renderFinishedShouldPresent(device),
      renderFinishedShouldSim(device)
{}
