//
// Created by samuel on 22/02/2021.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include "util/glm.h"

template<class T>
struct VulkanFormat;

template<>
struct VulkanFormat<glm::vec4> {
   constexpr static vk::Format Fmt = vk::Format::eR32G32B32A32Sfloat;
};

template<>
struct VulkanFormat<glm::vec2> {
    constexpr static vk::Format Fmt = vk::Format::eR32G32B32A32Sfloat;
};