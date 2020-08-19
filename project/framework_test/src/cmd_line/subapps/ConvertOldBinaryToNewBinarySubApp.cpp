//
// Created by samuel on 11/08/2020.
//

#include "ConvertOldBinaryToNewBinarySubApp.h"
#include <simulation/file_format/LegacySimDump.h>
#include <simulation/file_format/SimParams.h>
#include <simulation/file_format/SimSnapshot.h>

void ConvertOldBinaryToNewBinarySubApp::run() {
    auto inputDump = LegacySimDump::fromFile(inputPath);

    auto newSnapshot = SimSnapshot::from_legacy(inputDump);
    newSnapshot.to_file(outputPath);
}

void ConvertOldBinaryToNewBinarySubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("input_path", inputPath, "Binary file to convert")
            ->check(CLI::ExistingFile)
            ->required();

    subcommand->add_option("output_path", outputPath, "New binary file to output")
            ->required();
}
