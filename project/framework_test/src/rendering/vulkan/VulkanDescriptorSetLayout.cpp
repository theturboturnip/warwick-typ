//
// Created by samuel on 28/08/2020.
//

#include "VulkanDescriptorSetLayout.h"

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(vk::Device device, const std::vector<vk::DescriptorSetLayoutBinding>& bindings) {
    auto layoutInfo = vk::DescriptorSetLayoutCreateInfo{};
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();
    layout = device.createDescriptorSetLayoutUnique(layoutInfo);
}
