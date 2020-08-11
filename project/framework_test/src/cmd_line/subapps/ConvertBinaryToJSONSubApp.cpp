//
// Created by samuel on 11/08/2020.
//

#include "ConvertBinaryToJSONSubApp.h"
#include <simulation/file_format/LegacySimDump.h>
#include <simulation/file_format/SimParams.h>
#include <simulation/file_format/SimSnapshot.h>

void ConvertBinaryToJSONSubApp::run() {
    auto inputDump = LegacySimDump::fromFile(inputPath);

    auto newParams = SimParams::make_aca_default(
            Size<size_t>(inputDump.params.imax, inputDump.params.jmax),
            Size<float>(inputDump.params.xlength, inputDump.params.ylength)
    );
    auto newSnapshot = SimSnapshot::from_legacy(newParams, inputDump);
    std::ofstream output_file;
    output_file.open(outputPath);
    output_file << nlohmann::ordered_json(newSnapshot).dump(4);
}

void ConvertBinaryToJSONSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("input_path", inputPath, "Binary file to convert")
            ->check(CLI::ExistingFile)
            ->required();

    subcommand->add_option("output_path", outputPath, "JSON file to output")
            ->required();
}
