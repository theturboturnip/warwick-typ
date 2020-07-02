//
// Created by samuel on 02/07/2020.
//

#pragma once

#include "ISubApp.h"

class LegacyRenderPPMSubApp : public ISubApp {
public:
    ~LegacyRenderPPMSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "renderppm";
    }
    void setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) override;

    std::string inputFile;
    std::string outputFile;
    enum OutputMode {
        Zeta,
        Psi,
        Pressure
    };
    OutputMode outputMode;
};

