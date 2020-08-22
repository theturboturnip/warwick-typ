//
// Created by samuel on 22/08/2020.
//

#include "VulkanQueueFamilies.h"
VulkanQueueFamilies VulkanQueueFamilies::fill_from_vulkan(vk::PhysicalDevice device) {
    auto families = VulkanQueueFamilies();

    auto queueFamilyProperties = device.getQueueFamilyProperties();
    uint32_t queueIndex = 0;
    for (const auto& queueFamily : queueFamilyProperties) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            families.graphics_family = queueIndex;

        if (families.complete())
            break;

        queueIndex++;
    }

    return families;
}
