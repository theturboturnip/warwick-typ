//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "simulation/memory/SimulationAllocs.h"
#include "BaseVulkan2DAllocator.h"

#include <memory>

#include "simulation/file_format/SimSnapshot.h"

struct VulkanSimulationBuffers {
    vk::Buffer u, v, p, fluidmask;
};

template<class Allocator>
class VulkanSimulationAllocator {
    std::unique_ptr<BaseVulkan2DAllocator> alloc;

    template<typename T>
    auto allocMatrix(Size<uint32_t> matrixSize, const std::vector<T>& inputVec) {
        auto vulkanAlloc = alloc->allocateVulkan2D<T>(matrixSize);
        auto normalAlloc = alloc->mapFromVulkan(vulkanAlloc, &inputVec);

        return std::pair(vulkanAlloc.buffer, normalAlloc);
    };

public:
    VulkanSimulationAllocator(vk::Device device, vk::PhysicalDevice physicalDevice) : alloc(std::make_unique<Allocator>(device, physicalDevice)) {}


    std::pair<SimulationAllocs, VulkanSimulationBuffers> makeAllocs (const SimSnapshot& simSnapshot) {
        // TODO - Make a standard way of accessing padded size. It's weird to be constantly adding 2 to everything
        Size<uint32_t> matrixSize = {simSnapshot.simSize.pixel_size.x+2, simSnapshot.simSize.pixel_size.y+2};

        auto fluidmask_backing = std::vector<uint32_t>(simSnapshot.simSize.pixel_count());
        for (size_t i = 0; i < fluidmask_backing.size(); i++) {
            fluidmask_backing[i] = (simSnapshot.cell_type[i] == CellType::Fluid) ? 0xFFFFFFFF : 0;
        }

        auto [u_vulkan, u] = allocMatrix(matrixSize, simSnapshot.velocity_x);
        auto [v_vulkan, v] = allocMatrix(matrixSize, simSnapshot.velocity_y);
        auto [p_vulkan, p] = allocMatrix(matrixSize, simSnapshot.pressure);
        auto [fluidmask_vulkan, fluidmask] = allocMatrix(matrixSize, fluidmask_backing);

        const auto simAllocs = SimulationAllocs {
                .alloc = alloc.get(),
                .matrixSize = matrixSize,
                .u = u,
                .v = v,
                .p = p,
                .fluidmask = fluidmask
        };
        const auto vulkanAllocs = VulkanSimulationBuffers {
                .u = u_vulkan,
                .v = v_vulkan,
                .p = p_vulkan,
                .fluidmask = fluidmask_vulkan,
        };
        return std::pair(simAllocs, vulkanAllocs);
    }
};