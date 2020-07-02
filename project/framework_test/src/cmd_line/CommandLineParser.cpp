//
// Created by samuel on 02/07/2020.
//

#include "CommandLineParser.h"

#include <CLI/CLI11.hpp>

#include "cmd_line/subapps/FixedTimeSimSubApp.h"
#include "cmd_line/subapps/MakeInputSubApp.h"

int CommandLineParser::parseArguments(int argc, const char *argv[]) {
    CLI::App app{APP_NAME};

    // Require exectly one subcommand
    app.require_subcommand(1);
    app.failure_message(CLI::FailureMessage::help);
    // Whenever an argument is optional, print the default value
    app.option_defaults()->always_capture_default();

    std::vector<std::shared_ptr<ISubApp>> subapps =
            {{
                    std::make_shared<MakeInputSubApp>(),
                    std::make_shared<FixedTimeSimSubApp>(),
            }};

    const CommandLineConverters converters{};

    for (auto& subapp : subapps) {
        auto* subcommand = app.add_subcommand(subapp->cmdName());

        subapp->setupArgumentsForSubcommand(subcommand, converters);

        subcommand->callback([subapp](){ subapp->run(); });
    }

    CLI11_PARSE(app, argc, argv);
    return 0;
}
