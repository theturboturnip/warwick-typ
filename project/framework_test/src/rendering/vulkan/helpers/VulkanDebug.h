//
// Created by samuel on 08/02/2021.
//

#pragma once

#if !NDEBUG
constexpr static bool VulkanDebug = true;
#else
constexpr static bool VulkanDebug = false;
#endif