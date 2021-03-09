//
// Created by samuel on 02/07/2020.
//

#pragma once

#include "ISubApp.h"

#include "simulation/SimulationBackendEnum.h"

#include <optional>

class FixedTimeSimSubApp : public ISubApp {
public:
    ~FixedTimeSimSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "fixedtime";
    }
    void setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters& converters) override;

    std::string fluid_properties_file;
    std::string inputFile;
    float timeToRun;
    float maxFrequency;
    std::optional<std::string> outputFile;
    SimulationBackendEnum backend;
};
