//
// Created by samuel on 02/07/2020.
//

#pragma once

#include "ISubApp.h"

class CompareSubApp : public ISubApp {
public:
    ~CompareSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "compare";
    }
    void setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) override;

    std::string fileA;
    std::string fileB;
};

