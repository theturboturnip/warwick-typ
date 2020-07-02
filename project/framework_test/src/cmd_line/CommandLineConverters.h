//
// Created by samuel on 02/07/2020.
//

#pragma once

#include <CLI/CLI11.hpp>

#include "simulation/SimulationBackendEnum.h"

struct CommandLineConverters {
    CommandLineConverters();

    const std::map<std::string, SimulationBackendEnum> backendMap;
    const std::map<SimulationBackendEnum, std::string> backendToStrMap;
    const SimulationBackendEnum defaultBackend;

    void addBackendArgument(CLI::App* app, SimulationBackendEnum& backend) const;
};

// Generates a nice string->enum transformer given a mapping of string->enum.
//
// This is done instead of using a function returning IsMember, as I'm not entirely sure they can be stored by value?
// TODO: Check, and maybe remove this/replace it with a function.
#define ENUM_TRANSFORMER(Map) CLI::CheckedTransformer(Map, CLI::ignore_case)
