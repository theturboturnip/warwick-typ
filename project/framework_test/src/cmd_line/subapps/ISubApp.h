//
// Created by samuel on 02/07/2020.
//

#pragma once

#include <CLI/CLI11.hpp>

#include "cmd_line/CommandLineConverters.h"

#include <string>

/**
 *
 */
class ISubApp {
protected:
    ISubApp() = default;
public:
    ISubApp(const ISubApp&) = delete;
    virtual ~ISubApp() = default;

    virtual void run() = 0;

    virtual std::string cmdName() const = 0;
    virtual void setupArgumentsForSubcommand(CLI::App* subcommand, const CommandLineConverters& converters) = 0;
};