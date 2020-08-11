//
// Created by samuel on 11/08/2020.
//

#pragma once


#include "ISubApp.h"


class ConvertBinaryToJSONSubApp : public ISubApp {
public:
    ~ConvertBinaryToJSONSubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "convert2json";
    }
    void setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) override;

    std::string inputPath;
    std::string outputPath;
};
