//
// Created by samuel on 02/07/2020.
//

#pragma once

#include "ISubApp.h"

class MakeInputSubApp : public ISubApp {
public:
    MakeInputSubApp();
    ~MakeInputSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "makeinput";
    }
    void setupArgumentsForSubcommand(CLI::App* subcommand, const CommandLineConverters& converters) override;

    std::pair<float, float> physicalSize;
    std::string inputPath;
    std::string outputPath;
    bool constantVelocity;
    bool interpolatePressure;
};

