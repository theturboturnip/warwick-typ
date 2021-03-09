//
// Created by samuel on 02/07/2020.
//

#include "FixedTimeSimSubApp.h"

#include <chrono>
#include <simulation/file_format/FluidParams.h>

#include "simulation/file_format/SimSnapshot.h"
#include "simulation/runners/sim_fixedtime_runner/ISimFixedTimeRunner.h"

void FixedTimeSimSubApp::run() {
    auto fluid_props = FluidParams::from_file(fluid_properties_file);
    auto initial = SimSnapshot::from_file(inputFile);

    auto sim = ISimFixedTimeRunner::getForBackend(backend);

    auto start = std::chrono::steady_clock::now();

    auto output = sim->runForTime(fluid_props, initial, timeToRun);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    double timeTaken = diff.count();

    fprintf(stdout, "%f\n", timeTaken);
    fprintf(stderr, "performed calc in %f seconds\n", timeTaken);
    //fprintf(stderr, "enddump: %s", pretty_output_json.c_str());
    if (outputFile.has_value()){
        output.to_file(outputFile.value());
    }
}

void FixedTimeSimSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters& converters) {
    subcommand->add_option("fluid_properties", fluid_properties_file, "Fluid properties file")
            ->check(CLI::ExistingFile)
            ->required(true);
    subcommand->add_option("input_file", inputFile, "Initial simulation state file")
        ->check(CLI::ExistingFile)
        ->required(true);
    subcommand->add_option("time", timeToRun, "Time to run the simulation for")
        ->check(CLI::PositiveNumber)
        ->required(true);

    converters.addBackendArgument(subcommand, backend);

    subcommand->add_option("--output,-o", outputFile, "File to store the final simulation state");
}
