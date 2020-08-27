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

public:
    VulkanSimulationAllocator(vk::Device device, vk::PhysicalDevice physicalDevice) : alloc(std::make_unique<Allocator>(device, physicalDevice)) {}
    std::pair<SimulationAllocs, VulkanSimulationBuffers> makeAllocs (const SimSnapshot& simSnapshot) {
        // TODO - Make a standard way of accessing padded size. It's weird to be constantly adding 2 to everything
        Size<uint32_t> matrixSize = {simSnapshot.simSize.pixel_size.x+2, simSnapshot.simSize.pixel_size.y+2};

        auto fluidmask_backing = std::vector<uint32_t>(simSnapshot.simSize.pixel_count());
        for (size_t i = 0; i < fluidmask_backing.size(); i++) {
            fluidmask_backing[i] = (simSnapshot.cell_type[i] == CellType::Fluid) ? 0xFFFFFFFF : 0;
        }

        auto allocMatrix = [&](const auto& inputVec) {
            auto vulkanAlloc = alloc->allocateVulkan2D(matrixSize, &inputVec);
            auto normalAlloc = alloc->mapFromVulkan(vulkanAlloc);

            return std::pair(vulkanAlloc.buffer, normalAlloc);
        };

        auto [u_vulkan, u] = allocMatrix(simSnapshot.velocity_x);
        auto [v_vulkan, v] = allocMatrix(simSnapshot.velocity_y);
        auto [p_vulkan, p] = allocMatrix(simSnapshot.pressure);
        auto [fluidmask_vulkan, fluidmask] = allocMatrix(fluidmask_backing);

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