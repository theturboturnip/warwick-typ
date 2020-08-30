//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "simulation/memory/I2DAllocator.h"

#include <vulkan/vulkan.hpp>

/**
 * Abstract base class for vulkan-based allocators.
 *
 * The controller can allocate using allocAsVulkan() to get a VulkanMemory<T> struct. This can be mapped into an AllocatedMemory<T>.
 * Base classes must implement mapFromVulkan_unsafe(), to convert a VulkanMemory<void> into an AllocatedMemory<void> based on what platform is targeted.
 * i.e. a CPU-targeted allocator would use vkMapMemory, while a CUDA-targeted allocator would use CU_import/export.
 * Base classes must also handle unmapping memory inside freeAll (and MUST call their base class implementation AFTER they're done),
 * and must handle keeping track of mapped memory.
 * Base classes MUST also override the destructor with an implementation that calls *their* freeAll.
 *
 * VulkanMemory<T> has a vk::Buffer allocated with vk::DescriptorType::eStorageBuffer.
 */

// TODO - Ugh, organize the variable/type layout
class BaseVulkan2DAllocator : public I2DAllocator {
public:
    template<typename T>
    struct VulkanMemory {
        vk::DeviceMemory deviceMemory;
        vk::Buffer buffer;
        AllocatedMemory<T> unmappedMemoryInfo;

        template<typename U>
        VulkanMemory<U> unsafe_cast() const {
            return VulkanMemory<U> {
                .deviceMemory = deviceMemory,
                .buffer = buffer,
                .unmappedMemoryInfo = unmappedMemoryInfo.template unsafe_cast<U>()
            };
        }
    };

private:
    struct VulkanOwnedMemory {
        vk::UniqueDeviceMemory deviceMemory;
        vk::UniqueBuffer buffer;
    };
    std::vector<VulkanOwnedMemory> memories;
    vk::PhysicalDeviceMemoryProperties memProperties;
    vk::MemoryPropertyFlags expectedMemoryFlags;

    uint32_t selectMemoryTypeIndex(uint32_t memoryTypeBits);

    VulkanMemory<void> allocateVulkan_unsafe(Size<uint32_t> size, size_t elemSize);

protected:
    vk::Device device;

    BaseVulkan2DAllocator(const uint32_t usage, const vk::MemoryPropertyFlags expectedMemoryFlags, vk::Device device, vk::PhysicalDevice physicalDevice);

    virtual AllocatedMemory<void> mapFromVulkan_unsafe(VulkanMemory<void>, size_t elemSize, const void* initialData) = 0;
    AllocatedMemory<void> allocate2D_unsafe(Size<uint32_t> size, size_t elemSize, const void* initialData) override {
        VulkanMemory<void> vulkanMemory = allocateVulkan_unsafe(size, elemSize);
        return mapFromVulkan_unsafe(vulkanMemory, elemSize, initialData);
    }
public:

    template<typename T, typename = typename std::enable_if_t<!std::is_same_v<T, void>>>
    VulkanMemory<T> allocateVulkan2D(Size<uint32_t> size) {
        VulkanMemory<void> alloc = allocateVulkan_unsafe(size, sizeof(T));
        return alloc.unsafe_cast<T>();
    }

    // TODO - Adapt CPU versions to allow for changing backing pointers.
    //  If we want to pass it to Vulkan on the GPU, we'll have to unmap it at some point, and when we remap it won't be in the same place.
    template<typename T>
    AllocatedMemory<T> mapFromVulkan(const VulkanMemory<T>& vulkanMemory, const std::vector<T>* initialData) {
        if (initialData) {
            FATAL_ERROR_IF(vulkanMemory.unmappedMemoryInfo.totalSize != initialData->size(), "Expected size.x*y to be equal to initial data size\n");
        }
        auto typeErasedMapped = mapFromVulkan_unsafe(vulkanMemory.template unsafe_cast<void>(), sizeof(T), initialData ? initialData->data() : nullptr);
        return typeErasedMapped.template unsafe_cast<T>();
    }

    void freeAll() override;
    ~BaseVulkan2DAllocator() override;
};
