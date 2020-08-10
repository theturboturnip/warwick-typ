//
// Created by samuel on 02/07/2020.
//

#include "MakeInputSubApp.h"

#include "simulation/LegacySimulationParameters.h"
#include "simulation/file_format/LegacySimDump.h"
#include "util/fatal_error.h"

MakeInputSubApp::MakeInputSubApp() : exportType(ExportType::Empty) {}
void MakeInputSubApp::run() {
    LegacySimulationParameters params = {
            .imax = (int)resolution.first,
            .jmax = (int)resolution.second,
            .xlength = dimensions.first,
            .ylength = dimensions.second,
    };
    switch(exportType) {
        case ExportType::Empty:
            LegacySimDump(params).saveToFile(outputPath);
            break;
        default:
            FATAL_ERROR("Unimplemented exportType %d\n", exportType);
    }
}
void MakeInputSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters& converters) {
    std::map<std::string, ExportType> exportTypeMap = {{"empty", ExportType::Empty}};

    subcommand->add_option("output", outputPath, "Location of desired output file")->required();

    exportType = ExportType::Empty;
    subcommand->add_option("--type", exportType, "Type of input file to generate")
            ->transform(ENUM_TRANSFORMER(exportTypeMap));

    // TODO: These default values are unnecessary once we move away from ACA-based simulations
    resolution = {660, 120};
    dimensions = {22.0, 4.1};
    subcommand->add_option("--resolution", resolution, "Resolution in blocks of output file");
    subcommand->add_option("--dimensions", dimensions, "Dimensions in metres of output file");//->default_val(std::pair<float, float>{22.0, 4.1});
}
