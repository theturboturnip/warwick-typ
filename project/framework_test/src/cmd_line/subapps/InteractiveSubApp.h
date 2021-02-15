//
// Created by samuel on 22/08/2020.
//

#pragma once

#include "ISubApp.h"

#include "rendering/vulkan/SimAppProperties.h"

class InteractiveSubApp : public ISubApp {
public:
    ~InteractiveSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "run";
    }
    void setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters& converters) override;

    std::string fluid_properties_file;
    std::string input_file;
    std::optional<std::string> outputFile;
    SimulationBackendEnum backend;

    SimAppProperties simProperties;
};

