//
// Created by samuel on 27/08/2020.
//

#pragma once

#include "SimulationAllocs.h"

#include <memory>

#include "simulation/file_format/SimSnapshot.h"

template<class Allocator>
class BasicSimulationAllocator {
    std::unique_ptr<I2DAllocator> alloc;

public:
    BasicSimulationAllocator() : alloc(std::make_unique<Allocator>()) {}
    SimulationAllocs makeAllocs (const SimSnapshot& simSnapshot) {
        Size<uint32_t> matrixSize = simSnapshot.simSize.padded_pixel_size;

        auto fluidmask = std::vector<uint32_t>(simSnapshot.simSize.pixel_count());
        for (size_t i = 0; i < fluidmask.size(); i++) {
            fluidmask[i] = (simSnapshot.cell_type[i] == CellType::Fluid) ? 0xFFFFFFFF : 0;
        }

        return SimulationAllocs {
                .alloc = alloc.get(),
                .matrixSize = matrixSize,
                .u = alloc->allocate2D(matrixSize, &simSnapshot.velocity_x),
                .v = alloc->allocate2D(matrixSize, &simSnapshot.velocity_y),
                .p = alloc->allocate2D(matrixSize, &simSnapshot.pressure),
                .fluidmask = alloc->allocate2D(matrixSize, &fluidmask)
        };
    }
};