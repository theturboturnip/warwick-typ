//
// Created by samuel on 09/08/2020.
//
#include "SimParams.h"

#include <nlohmann/json.hpp>


//SimParams SimParams::build_from_json(nlohmann::json json_object) {
//
//
//    return SimParams{
//            .pixel_size = Size(0, 0),
//            .physical_size = Size(0, 0),
//    };
//}

void to_json(nlohmann::json& j, const SimParams& p){
    j = nlohmann::json{};
    j["pixel_size"]["x"] = p.pixel_size.x;
    j["pixel_size"]["y"] = p.pixel_size.y;
    j["physical_size"]["x"] = p.physical_size.x;
    j["physical_size"]["y"] = p.physical_size.y;
    j["fluid"]["Re"] = p.fluid.Re;
    j["sim"]["timestep_divisor"] = p.sim.timestep_divisor;
    j["sim"]["max_timestep_divisor"] = p.sim.max_timestep_divisor;
    j["sim"]["timestep_safety"] = p.sim.timestep_safety;
    j["sim"]["gamma"] = p.sim.gamma;
    j["sim"]["redblack_poisson"]["max_iterations"] = p.sim.redblack_poisson.max_iterations;
    j["sim"]["redblack_poisson"]["error_threshold"] = p.sim.redblack_poisson.error_threshold;
    j["sim"]["redblack_poisson"]["omega"] = p.sim.redblack_poisson.omega;
}

void from_json(const nlohmann::json &j, SimParams &p) {
    j.at("pixel_size").at("x").get_to(p.pixel_size.x);
    j.at("pixel_size").at("y").get_to(p.pixel_size.y);
    j.at("physical_size").at("x").get_to(p.physical_size.x);
    j.at("physical_size").at("y").get_to(p.physical_size.y);

    j.at("fluid").at("Re").get_to(p.fluid.Re);
    j.at("sim").at("timestep_divisor").get_to(p.sim.timestep_divisor);
    j.at("sim").at("max_timestep_divisor").get_to(p.sim.max_timestep_divisor);
    j.at("sim").at("timestep_safety").get_to(p.sim.timestep_safety);
    j.at("sim").at("gamma").get_to(p.sim.gamma);
    j.at("sim").at("redblack_poisson").at("max_iterations").get_to(p.sim.redblack_poisson.max_iterations);
    j.at("sim").at("redblack_poisson").at("error_threshold").get_to(p.sim.redblack_poisson.error_threshold);
    j.at("sim").at("redblack_poisson").at("omega").get_to(p.sim.redblack_poisson.omega);
}