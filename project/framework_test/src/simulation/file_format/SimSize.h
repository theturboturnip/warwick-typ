//
// Created by samuel on 19/08/2020.
//

#pragma once

#include "LegacySimSize.h"
#include "util/Size.h"

struct SimSize {
    SimSize(Size<size_t> pixel_size, Size<float> physical_size)
        : pixel_size(pixel_size),
          physical_size(physical_size)
    {}
    static SimSize from_legacy(LegacySimSize legacy) {
        return SimSize(
                {legacy.imax, legacy.jmax},
                {legacy.xlength, legacy.ylength}
        );
    }

    // Size of the simulation grid in pixels
    const Size<size_t> pixel_size;
    // Size of the simulation grid in meters
    const Size<float> physical_size;

    [[nodiscard]] inline size_t pixel_count() const {
        return (pixel_size.x+2) * (pixel_size.y+2);
    }

    // Meters/pixel for x and y
    [[nodiscard]] inline float del_x() const {
        return physical_size.x/pixel_size.x;
    }
    [[nodiscard]] inline float del_y() const {
        return physical_size.y/pixel_size.y;
    }

    [[nodiscard]] LegacySimSize to_legacy() const {
        return LegacySimSize {
                .imax=pixel_size.x,
                .jmax=pixel_size.y,
                .xlength=physical_size.x,
                .ylength=physical_size.y,
        };
    }
};
