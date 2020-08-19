//
// Created by samuel on 09/08/2020.
//

#pragma once

#include <cstdint>

#include "LegacySimDump.h"
#include "util/Size.h"

enum class CellType : uint8_t {
    Boundary,
    Fluid
};

struct SimSnapshot {
    explicit SimSnapshot(Size<size_t> pixel_size, Size<float> physical_size);
    static SimSnapshot from_legacy(const LegacySimDump& from_legacy_dump);
    static SimSnapshot from_file(std::string path);
    void to_file(std::string path) const;

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

    // TODO - Allow this to be templated on float/double?
    std::vector<float> velocity_x;
    std::vector<float> velocity_y;
    std::vector<float> pressure;

    std::vector<CellType> cell_type;

    [[nodiscard]] std::vector<char> get_legacy_cell_flags() const;
    [[nodiscard]] int get_boundary_cell_count() const;

    [[nodiscard]] static std::vector<CellType> cell_type_from_legacy(const std::vector<char> legacyFlags);

    [[nodiscard]] LegacySimulationParameters params_to_legacy() const;
    [[nodiscard]] LegacySimDump to_legacy() const;
};