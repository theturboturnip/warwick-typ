//
// Created by samuel on 09/08/2020.
//
#include "SimSnapshot.h"

#include "util/fatal_error.h"

SimSnapshot::SimSnapshot(SimSize simSize)
    : simSize(simSize),
      velocity_x(simSize.pixel_count(), 0.0),
      velocity_y(simSize.pixel_count(), 0.0),
      pressure(simSize.pixel_count(), 0.0),
      cell_type(simSize.pixel_count(), CellType::Fluid)
{}

LegacySimDump SimSnapshot::to_legacy() const {
    LegacySimSize legacy_params = simSize.to_legacy();

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
    auto legacy = std::vector<char>(simSize.pixel_count(), 0);

    const size_t width = simSize.padded_pixel_size.x;
    const size_t height = simSize.padded_pixel_size.y;
    for (size_t i = 0; i < width; ++i) {
        for (size_t j = 0; j < height; ++j) {
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
SimSnapshot SimSnapshot::from_legacy(const LegacySimDump &from_legacy_dump) {
    auto snapshot = SimSnapshot(SimSize::from_legacy(from_legacy_dump.simSize));

    snapshot.velocity_x = from_legacy_dump.u;
    snapshot.velocity_y = from_legacy_dump.v;
    snapshot.pressure = from_legacy_dump.p;

    snapshot.cell_type = cell_type_from_legacy(from_legacy_dump.flag);

    return snapshot;
}
SimSnapshot SimSnapshot::from_file(std::string path) {
    // TODO - implement this without Legacy data
    auto legacy = LegacySimDump::fromFile(path);
    return SimSnapshot::from_legacy(legacy);
}
void SimSnapshot::to_file(std::string path) const {
    to_legacy().saveToFile(path);
}
std::vector<CellType> SimSnapshot::cell_type_from_legacy(const std::vector<char> legacyFlags) {
    std::vector<CellType> cell_type(legacyFlags.size(), CellType::Fluid);
    for (size_t i = 0; i < legacyFlags.size(); i++) {
        if (legacyFlags[i] & C_F)
            cell_type[i] = CellType::Fluid;
        else
            cell_type[i] = CellType::Boundary;
    }
    return cell_type;
}
