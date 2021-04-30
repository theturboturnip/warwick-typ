//
// Created by samuel on 18/04/2021.
//

#pragma once


#include "ISubApp.h"

class ResidualSubApp : public ISubApp {
public:
    ResidualSubApp();
    ~ResidualSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "residual";
    }
    void setupArgumentsForSubcommand(CLI::App* subcommand, const CommandLineConverters& converters) override;

    std::string inputPath;
    std::string fluidPropertiesFile;
};

