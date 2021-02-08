//
// Created by samuel on 22/08/2020.
//

#include "VulkanQueueFamilies.h"
std::optional<VulkanQueueFamilies> VulkanQueueFamilies::getForDevice(vk::PhysicalDevice device, vk::UniqueSurfaceKHR& surface) {
    auto queueFamilyProperties = device.getQueueFamilyProperties();
    uint32_t queueFamilyIndex = 0;

    std::optional<uint32_t> graphicsFamily = std::nullopt;
    std::optional<uint32_t> presentFamily = std::nullopt;

    for (const auto& queueFamily : queueFamilyProperties) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            graphicsFamily = queueFamilyIndex;

        if (device.getSurfaceSupportKHR(queueFamilyIndex, *surface)) {
            presentFamily = queueFamilyIndex;
        }

        if (graphicsFamily.has_value() && presentFamily.has_value()) {
            return VulkanQueueFamilies{
                    .graphicsFamily = graphicsFamily.value(),
                    .presentFamily = presentFamily.value()
            };
        }

        queueFamilyIndex++;
    }

    // We didn't find both a graphics and present family, we failed.
    return std::nullopt;
}
