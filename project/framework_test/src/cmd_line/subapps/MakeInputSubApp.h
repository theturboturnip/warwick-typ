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
    std::string cmdName() const override {
        return "makeinput";
    }
    void setupArgumentsForSubcommand(CLI::App* subcommand) override;

    enum ExportType {
        Empty
    };
    ExportType exportType;
    std::pair<size_t, size_t> resolution;
    std::pair<float, float> dimensions;
    std::string outputPath;
};

