//
// Created by samuel on 02/07/2020.
//

#include "FixedTimeSimSubApp.h"

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/runners/sim_10s_runner/ISim10sRunner.h"

#include <chrono>

struct SimRunData {
    LegacySimDump finalDump;
    double timeInSeconds = 0;
};

SimRunData runSimWithFixedTimeRunner(SimulationBackendEnum backend, const LegacySimDump& initial) {
    auto sim = ISim10sRunner::getForBackend(backend);

    auto start = std::chrono::steady_clock::now();

    const float baseTimestep = 1.0f/30.0f;
    LegacySimDump output = sim->runFor10s(initial, baseTimestep);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;

    return SimRunData{ .finalDump = std::move(output), .timeInSeconds = diff.count() };
}

void FixedTimeSimSubApp::run() {
    LegacySimDump initial = LegacySimDump::fromFile(inputFile);

    // TODO: Use the fixed time
    SimRunData runData = runSimWithFixedTimeRunner(backend, initial);

    LegacySimDump& output = runData.finalDump;
    fprintf(stderr, "performed calc in %f seconds\n", runData.timeInSeconds);
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
