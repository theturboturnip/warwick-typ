//
// Created by samuel on 09/08/2020.
//

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <nlohmann/adl_serializer.hpp>

#include "LegacySimDump.h"
#include "SimParams.h"

enum class CellType : uint8_t {
    Boundary,
    Fluid
};

struct SimSnapshot {
    explicit SimSnapshot(const SimParams& params);
    static SimSnapshot from_legacy(const SimParams& params, const LegacySimDump& from_legacy_dump);

    const SimParams params;

    // TODO - Allow this to be templated on float/double?
    std::vector<float> velocity_x;
    std::vector<float> velocity_y;
    std::vector<float> pressure;

    std::vector<CellType> cell_type;

    [[nodiscard]] std::vector<uint8_t> get_legacy_cell_flags() const;
    [[nodiscard]] int get_boundary_cell_count() const;

    [[nodiscard]] LegacySimDump to_legacy() const;
};

// SimSnapshot has a const field, which means you can't use the normal to_json/from_json functions.
// You have to implement a serializer in the nohlmann namespace
namespace nlohmann {
    template<>
    struct adl_serializer<SimSnapshot> {
        static SimSnapshot from_json(const nlohmann::json& j);
        static void to_json(nlohmann::json& j, const SimSnapshot& s);
    };
}