//
// Created by samuel on 02/07/2020.
//

#include "MakeInputSubApp.h"

#include "util/fatal_error.h"
#include <simulation/file_format/SimSize.h>
#include <simulation/file_format/SimSnapshot.h>
#include <stb_image.h>

MakeInputSubApp::MakeInputSubApp() {}
void MakeInputSubApp::run() {
    int width, height, channels;
    uint8_t* data = stbi_load(inputPath.c_str(), &width, &height, &channels, 3);
    if (!data) {
        FATAL_ERROR("Loading image %s failed: %s\n", inputPath.c_str(), stbi_failure_reason());
    }

    if (channels == 4) {
        printf("WARNING: Input image %s had 4 channels originally, but this parser only supports 3.\n"
               "Image will be parsed ignoring the alpha channel.\n",
               inputPath.c_str());
    }

    if (width <= 2 || height <= 2) {
        FATAL_ERROR("Image %s is %dx%d, but must be >2 in each direction\n", inputPath.c_str(), width, height);
    }

    // NOTE - Input produced includes the boundary padding
    auto size = SimSize(
            {(uint32_t) width-2, (uint32_t) height-2},
            physicalSize
    );
    auto simSnapshot = SimSnapshot(size);
    // stb coordinates are top-left (0,0), bottom-right (width-1, height-1)
    // our coordinates are top-left (0, height-1), bottom-right (width-1,0)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            const size_t idx = (x * height) + (height - 1 - y);

            const uint8_t red = data[(y * width + x)*3 + 0];
            const uint8_t green = data[(y * width + x)*3 + 1];
            const uint8_t blue = data[(y * width + x)*3 + 2];

            if (red == 0 && green == 0 && blue == 0) {
                simSnapshot.cell_type[idx] = CellType::Fluid;
            } else {
                simSnapshot.cell_type[idx] = CellType::Boundary;
            }
            simSnapshot.velocity_x[idx] = 0;
            simSnapshot.velocity_y[idx] = 0;
            simSnapshot.pressure[idx] = 0;
        }
    }

    stbi_image_free(data);

    simSnapshot.to_file(outputPath);
}
void MakeInputSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters& converters) {
    subcommand->add_option("input", inputPath, "Input image - black squares are fluid, nonblack squares are obstacle.")
            ->check(CLI::ExistingFile)
            ->required();
    subcommand->add_option("physical-size", physicalSize, "Resolution in metres of output file")->required();
    subcommand->add_option("output", outputPath, "Output file")->required();
}
