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

    std::string inputFile;
    float timeToRun;
    std::optional<std::string> outputFile;
    SimulationBackendEnum backend;
};
