//
// Created by samuel on 11/08/2020.
//

#pragma once


#include "ISubApp.h"


class ConvertOldBinaryToNewBinarySubApp : public ISubApp {
public:
    ~ConvertOldBinaryToNewBinarySubApp() override = default;

    void run() override;

    [[nodiscard]] std::string cmdName() const override {
        return "convert2newbinary";
    }
    void setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) override;

    std::string inputPath;
    std::string outputPath;
};
