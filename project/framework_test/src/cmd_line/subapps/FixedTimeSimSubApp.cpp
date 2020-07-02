//
// Created by samuel on 02/07/2020.
//

#include "FixedTimeSimSubApp.h"

#include "simulation/file_format/LegacySimDump.h"

#include <chrono>
#include "simulation/runners/sim_fixedtime_runner/ISimFixedTimeRunner.h"

void FixedTimeSimSubApp::run() {
    LegacySimDump initial = LegacySimDump::fromFile(inputFile);

    auto sim = ISimFixedTimeRunner::getForBackend(backend);

    LegacySimDump output;
    double timeTaken;
    const float baseTimestep = 1.0f / 30.0f;
    {
        auto start = std::chrono::steady_clock::now();

        output = sim->runForTime(initial, baseTimestep, timeToRun);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        timeTaken = diff.count();
    }

    fprintf(stderr, "performed calc in %f seconds\n", timeTaken);
    fprintf(stderr, "enddump: %s", output.debugString().c_str());
    if (outputFile.has_value())
        output.saveToFile(outputFile.value());
}

void FixedTimeSimSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters& converters) {
    subcommand->add_option("input_file", inputFile, "Initial simulation state file")
        ->check(CLI::ExistingFile)
        ->required(true);
    subcommand->add_option("time", timeToRun, "Time to run the simulation for")
        ->check(CLI::PositiveNumber)
        ->required(true);

    converters.addBackendArgument(subcommand, backend);

    subcommand->add_option("--output,-o", outputFile, "File to store the final simulation state");
}
