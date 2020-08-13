//
// Created by samuel on 09/08/2020.
//
#include "SimSnapshot.h"

#include "util/fatal_error.h"

SimSnapshot::SimSnapshot(const SimParams &params)
    : params(params),
      velocity_x(params.pixel_count(), 0.0),
      velocity_y(params.pixel_count(), 0.0),
      pressure(params.pixel_count(), 0.0),
      cell_type(params.pixel_count(), CellType::Fluid)
{
}

LegacySimDump SimSnapshot::to_legacy() const {
    LegacySimulationParameters legacy_params = params.to_legacy();

    auto dump = LegacySimDump(legacy_params);

    dump.u = velocity_x;
    dump.v = velocity_y;
    dump.p = pressure;
    dump.flag = get_legacy_cell_flags();

    return dump;
}

int SimSnapshot::get_boundary_cell_count() const {
    int boundary_cell_count = 0;
    for (CellType cell : cell_type) {
        if (cell == CellType::Boundary)
            boundary_cell_count += 1;
    }
    return boundary_cell_count;
}

std::vector<char> SimSnapshot::get_legacy_cell_flags() const {
    auto legacy = std::vector<char>(params.pixel_count(), 0);

    const int width = params.pixel_size.x+2;
    const int height = params.pixel_size.y+2;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            int pixel_idx = i * height + j;
            if (cell_type[pixel_idx] == CellType::Fluid) {
                legacy[pixel_idx] = C_F;
            } else {
                legacy[pixel_idx] = C_B;

                // If j != height - 1, there exists a cell to the north.
                // If it's fluid, set the flag for "this boundary cell has fluid to the north"
                if ((j != height - 1) && (cell_type[i * height + (j + 1)] == CellType::Fluid)) {
                    legacy[pixel_idx] |= B_N;
                }
                // if j != 0, there's a cell to the south at j - 1
                if ((j != 0) && (cell_type[i * height + (j - 1)] == CellType::Fluid)) {
                    legacy[pixel_idx] |= B_S;
                }
                // if i != width - 1, there's a cell to the east at i + 1
                if ((i != width - 1) && (cell_type[(i + 1) * height + j] == CellType::Fluid)) {
                    legacy[pixel_idx] |= B_E;
                }
                // if i != 0, there's a cell to the west at i - 1
                if ((i != 0) && (cell_type[(i - 1) * height + j] == CellType::Fluid)) {
                    legacy[pixel_idx] |= B_W;
                }
            }


        }
    }

    return legacy;
}
SimSnapshot SimSnapshot::from_legacy(const SimParams& params, const LegacySimDump &from_legacy_dump) {
    auto snapshot = SimSnapshot(params);

    snapshot.velocity_x = from_legacy_dump.u;
    snapshot.velocity_y = from_legacy_dump.v;
    snapshot.pressure = from_legacy_dump.p;

    for (size_t i = 0; i < from_legacy_dump.flag.size(); i++) {
        if (from_legacy_dump.flag[i] & C_F)
            snapshot.cell_type[i] = CellType::Fluid;
        else
            snapshot.cell_type[i] = CellType::Boundary;
    }

    return snapshot;
}
SimSnapshot SimSnapshot::from_file(std::string path) {
    nlohmann::json j;
    auto input = std::ifstream(path);
    input >> j;
    return j.get<SimSnapshot>();
}

namespace nlohmann{
    void adl_serializer<SimSnapshot>::to_json(ordered_json& j, const SimSnapshot& s) {
        j["params"] = s.params;
        j["velocity_x"] = nlohmann::json(s.velocity_x).dump();
        j["velocity_y"] = nlohmann::json(s.velocity_y).dump();
        j["pressure"] = nlohmann::json(s.pressure).dump();
        j["cell_type"] = nlohmann::json(s.cell_type).dump();
    }

    SimSnapshot adl_serializer<SimSnapshot>::from_json(const ordered_json &j) {
        SimParams params{};
        j.at("params").get_to(params);

        auto snapshot = SimSnapshot(params);

        auto extract_vector = [&j, &params](std::string key, auto& vec_ref) {
            std::string array_dump;
            j.at(key).get_to(array_dump);
            auto array_json = nlohmann::json::parse(array_dump);
            array_json.get_to(vec_ref);
            DASSERT_M(vec_ref.size() == params.pixel_count(), "Size mismatch for %s - expected %zu got %zu\n", key.c_str(), params.pixel_count(), array_json.size());
        };

        extract_vector("velocity_x", snapshot.velocity_x);
        extract_vector("velocity_y", snapshot.velocity_y);
        extract_vector("pressure", snapshot.pressure);
        extract_vector("cell_type", snapshot.cell_type);

        return snapshot;
    }
}
