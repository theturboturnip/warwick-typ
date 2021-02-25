//
// Created by samuel on 22/08/2020.
//

#include "VulkanQueueFamilies.h"
std::optional<VulkanQueueFamilies> VulkanQueueFamilies::getForDevice(vk::PhysicalDevice device, vk::UniqueSurfaceKHR& surface) {
    std::optional<uint32_t> graphicsFamily = std::nullopt;
    std::optional<uint32_t> presentFamily = std::nullopt;
    std::optional<uint32_t> computeFamily = std::nullopt;

    auto queueFamilyProperties = device.getQueueFamilyProperties();
    uint32_t queueFamilyIndex = 0;
    for (const auto& queueFamily : queueFamilyProperties) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            graphicsFamily = queueFamilyIndex;
        }

        if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute && !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
            computeFamily = queueFamilyIndex;
        }

        if (device.getSurfaceSupportKHR(queueFamilyIndex, *surface)) {
            presentFamily = queueFamilyIndex;
        }

        if (graphicsFamily.has_value() &&
            presentFamily.has_value() &&
            computeFamily.has_value()) {
            return VulkanQueueFamilies{
                    .graphicsFamily = graphicsFamily.value(),
                    .presentFamily = presentFamily.value(),
                    .computeFamily = computeFamily.value()
            };
        }

        queueFamilyIndex++;
    }

    // We didn't find both a graphics and present family, we failed.
    return std::nullopt;
}
