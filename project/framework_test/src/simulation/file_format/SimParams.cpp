//
// Created by samuel on 09/08/2020.
//
#include "SimParams.h"

#include <nlohmann/json.hpp>


void to_json(nlohmann::json& j, const SimParams& p){
    j = nlohmann::json{};
    j["pixel_size"]["x"] = p.pixel_size.x;
    j["pixel_size"]["y"] = p.pixel_size.y;
    j["physical_size"]["x"] = p.physical_size.x;
    j["physical_size"]["y"] = p.physical_size.y;
    j["Re"] = p.Re;
    j["initial_velocity_x"] = p.initial_velocity_x;
    j["initial_velocity_y"] = p.initial_velocity_y;
    j["timestep_divisor"] = p.timestep_divisor;
    j["max_timestep_divisor"] = p.max_timestep_divisor;
    j["timestep_safety"] = p.timestep_safety;
    j["gamma"] = p.gamma;
    j["poisson_max_iterations"] = p.poisson_max_iterations;
    j["poisson_error_threshold"] = p.poisson_error_threshold;
    j["poisson_omega"] = p.poisson_omega;
}

void from_json(const nlohmann::json &j, SimParams &p) {
    j.at("pixel_size").at("x").get_to(p.pixel_size.x);
    j.at("pixel_size").at("y").get_to(p.pixel_size.y);
    j.at("physical_size").at("x").get_to(p.physical_size.x);
    j.at("physical_size").at("y").get_to(p.physical_size.y);

    j.at("Re").get_to(p.Re);
    j.at("initial_velocity_x").get_to(p.initial_velocity_x);
    j.at("initial_velocity_y").get_to(p.initial_velocity_y);
    j.at("timestep_divisor").get_to(p.timestep_divisor);
    j.at("max_timestep_divisor").get_to(p.max_timestep_divisor);
    j.at("timestep_safety").get_to(p.timestep_safety);
    j.at("gamma").get_to(p.gamma);
    j.at("poisson_max_iterations").get_to(p.poisson_max_iterations);
    j.at("poisson_error_threshold").get_to(p.poisson_error_threshold);
    j.at("poisson_omega").get_to(p.poisson_omega);
}

LegacySimulationParameters SimParams::to_legacy() const {
    return LegacySimulationParameters {
            .imax=int(pixel_size.x),
            .jmax=int(pixel_size.y),

            .xlength=physical_size.x,
            .ylength=physical_size.y,
    };
}
SimParams SimParams::make_aca_default(Size<size_t> pixel_size, Size<float> physical_size) {
    return SimParams{
            .pixel_size = pixel_size,
            .physical_size = physical_size,

            .Re = 150.0f,
            .initial_velocity_x = 1.0,
            .initial_velocity_y = 0.0,

            .timestep_divisor = 60,
            .max_timestep_divisor = 480,
            .timestep_safety = 0.5f,

            .gamma = 0.9f,

            .poisson_max_iterations = 100,
            .poisson_error_threshold = 0.001f,
            .poisson_omega = 1.7f,
    };
}
/*SimParams::SimParams(const LegacySimulationParameters &from_legacy, int timestep_divisor, int max_timestep_divisor)
    : pixel_size(from_legacy.imax, from_legacy.jmax),
      physical_size(from_legacy.xlength, from_legacy.ylength),

      fluid({
              .Re = 150.0f,
              .initial_velocity_x = 1.0,
              .initial_velocity_y = 0.0,
      }),

      sim({
              .timestep_divisor = timestep_divisor,
              .max_timestep_divisor = max_timestep_divisor,
              .timestep_safety = 0.5f,

              .gamma = 0.9f,

              .redblack_poisson = {
                      .max_iterations = 100,
                      .error_threshold = 0.001f,
                      .omega = 1.7f
              }
      })
{
}
*/