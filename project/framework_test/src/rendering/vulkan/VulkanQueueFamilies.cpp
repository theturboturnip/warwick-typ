//
// Created by samuel on 22/08/2020.
//

#include "VulkanQueueFamilies.h"
VulkanQueueFamilies VulkanQueueFamilies::fill_from_vulkan(vk::PhysicalDevice device, vk::UniqueSurfaceKHR& surface) {
    auto families = VulkanQueueFamilies();

    auto queueFamilyProperties = device.getQueueFamilyProperties();
    uint32_t queueFamilyIndex = 0;
    for (const auto& queueFamily : queueFamilyProperties) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            families.graphics_family = queueFamilyIndex;

        if (device.getSurfaceSupportKHR(queueFamilyIndex, *surface)) {
            families.present_family = queueFamilyIndex;
        }

        if (families.complete())
            break;

        queueFamilyIndex++;
    }

    return families;
}
