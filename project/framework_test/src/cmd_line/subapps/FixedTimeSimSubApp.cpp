//
// Created by samuel on 02/07/2020.
//

#include "FixedTimeSimSubApp.h"

#include <chrono>

#include "simulation/file_format/SimSnapshot.h"
#include "simulation/runners/sim_fixedtime_runner/ISimFixedTimeRunner.h"

void FixedTimeSimSubApp::run() {
    nlohmann::json initial_json;
    std::ifstream(inputFile) >> initial_json;
    auto initial = initial_json.get<SimSnapshot>();

    auto sim = ISimFixedTimeRunner::getForBackend(backend);

    double timeTaken;

    auto start = std::chrono::steady_clock::now();

    auto output = sim->runForTime(initial, timeToRun);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    timeTaken = diff.count();

    nlohmann::ordered_json output_json = output;
    std::string pretty_output_json = output_json.dump(4);
    fprintf(stderr, "performed calc in %f seconds\n", timeTaken);
    //fprintf(stderr, "enddump: %s", pretty_output_json.c_str());
    if (outputFile.has_value()){
        std::ofstream output_stream;
        output_stream.open(outputFile.value());
        output_stream << pretty_output_json;
    }
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
