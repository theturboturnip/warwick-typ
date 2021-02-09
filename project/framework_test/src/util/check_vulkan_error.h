//
// Created by samuel on 09/02/2021.
//

#pragma once

#define VULKAN_CHECK_IN_RELEASE 1
#if !NDEBUG
#define VULKAN_CHECK_ENABLED 1
#elif VULKAN_CHECK_IN_RELEASE
#define VULKAN_CHECK_ENABLED 1
#else
#define VULKAN_CHECK_ENABLED 0
#endif

#if VULKAN_CHECK_ENABLED
#include "fatal_error.h"

#define CHECK_VULKAN_ERROR(error) do { FATAL_ERROR_IF(error != vk::Result::eSuccess, "Vulkan Error: %s\n", vk::to_string(error).c_str()); } while(0);
#define CHECKED_VULKAN(X) ([&]{ vk::Result error = (vk::Result)(X); CHECK_VULKAN_ERROR(error); }())
#else
#define CHECK_VULKAN_ERROR(error)
#define CHECKED_VULKAN(X) (X)
#endif
