//
// Created by samuel on 09/08/2020.
//

#pragma once

#include <cstdint>

#include "LegacySimDump.h"
#include "SimSize.h"
#include "util/Size.h"

enum class CellType : uint8_t {
    Boundary,
    Fluid
};

struct SimSnapshot {
    explicit SimSnapshot(SimSize simSize);
    static SimSnapshot from_legacy(const LegacySimDump& from_legacy_dump);
    static SimSnapshot from_file(std::string path);
    void to_file(std::string path) const;

    const SimSize simSize;

    // TODO - Allow this to be templated on float/double?
    std::vector<float> velocity_x;
    std::vector<float> velocity_y;
    std::vector<float> pressure;

    std::vector<CellType> cell_type;

    [[nodiscard]] std::vector<char> get_legacy_cell_flags() const;
    [[nodiscard]] int get_boundary_cell_count() const;

    [[nodiscard]] static std::vector<CellType> cell_type_from_legacy(const std::vector<char> legacyFlags);

    [[nodiscard]] LegacySimDump to_legacy() const;
};