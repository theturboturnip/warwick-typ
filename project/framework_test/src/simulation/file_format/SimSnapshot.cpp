//
// Created by samuel on 09/08/2020.
//
#include "SimSnapshot.h"

SimSnapshot::SimSnapshot(const SimParams &params)
    : params(params),
      velocity_x(params.pixel_size.x*params.pixel_size.y, 0.0),
      velocity_y(params.pixel_size.x*params.pixel_size.y, 0.0),
      pressure(params.pixel_size.x*params.pixel_size.y, 0.0),
      cell_type(params.pixel_size.x*params.pixel_size.y, CellType::Fluid)
{
}
LegacySimDump SimSnapshot::to_legacy() {
    LegacySimulationParameters legacy_params{
            .imax=params.pixel_size.x,
            .jmax=params.pixel_size.y,

            .xlength=params.physical_size.x,
            .ylength=params.physical_size.y,
    };

    auto dump = LegacySimDump(legacy_params);

    dump.u = velocity_x;
    dump.v = velocity_y;
    dump.p = pressure;

    // TODO - flags

    return dump;
}

namespace nlohmann{
    void adl_serializer<SimSnapshot>::to_json(nlohmann::json& j, const SimSnapshot& s) {
        j["params"] = s.params;
        j["velocity_x"] = s.velocity_x;
        j["velocity_y"] = s.velocity_y;
        j["pressure"] = s.pressure;
        j["cell_type"] = s.cell_type;
    }

    SimSnapshot adl_serializer<SimSnapshot>::from_json(const json &j) {
        SimParams params{};
        j.at("params").get_to(params);

        auto snapshot = SimSnapshot(params);

        if (j.at("velocity_x").size() != snapshot.velocity_x.size())
            throw std::runtime_error("Velocity X size mismatch");
        j.at("velocity_x").get_to(snapshot.velocity_x);

        if (j.at("velocity_y").size() != snapshot.velocity_y.size())
            throw std::runtime_error("Velocity Y size mismatch");
        j.at("velocity_y").get_to(snapshot.velocity_y);

        if (j.at("pressure").size() != snapshot.pressure.size())
            throw std::runtime_error("Pressure size mismatch");
        j.at("pressure").get_to(snapshot.pressure);

        if (j.at("cell_type").size() != snapshot.cell_type.size())
            throw std::runtime_error("Cell Type size mismatch");
        j.at("cell_type").get_to(snapshot.cell_type);

        return snapshot;
    }
}
