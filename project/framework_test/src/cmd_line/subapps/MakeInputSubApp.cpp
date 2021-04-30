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
    fprintf(stdout, "Loading image %s: size = %d x %d\n", inputPath.c_str(), width, height);

    if (channels == 4) {
        printf("WARNING: Input image %s had 4 channels originally, but this parser only supports 3.\n"
               "Image will be parsed ignoring the alpha channel.\n",
               inputPath.c_str());
    }

    if (width <= 2 || height <= 2) {
        FATAL_ERROR("Image %s is %dx%d, but must be >2 in each direction\n", inputPath.c_str(), width, height);
    }

    if (constantVelocity) {
        // Check the constant velocity produces a sensible timestep
        float velocityX = 1;
        float cellSizeX = physicalSize.first * 1.0f / width;
        // for maximum possible delT: delT * velocityX = cellSizeX
        float maxDelT = cellSizeX / velocityX;
        fprintf(stdout, "Velocity X = %g, cell size X = %g\nMaximum delta T = %g (%d Hz)\n", velocityX, cellSizeX, maxDelT, (int)(1.0f/maxDelT));
    }

    // Image should not include boundary padding - it's just a silhouette
    auto size = SimSize(
            {(uint32_t) width, (uint32_t) height},
            physicalSize
    );
    auto simSnapshot = SimSnapshot(size);

    auto paddedSize = size.padded_pixel_size;
    // For every pixel in the actual image (which doesn't include padding), set the values
    // stb coordinates are top-left (0,0), bottom-right (width-1, height-1)
    // our coordinates are top-left (0, height-1), bottom-right (width-1,0)
    std::vector<int> occluded(height,0);
    // As we're going from left(0) to right(width), we can tell if there's something between us and the left by storing a column of "occluded" values
    // then when we encounter an occluder, set it to true for that row.
    // if occluded[y] == 1 then there was for some x < current_x a pixel[x,y] which was an occluder.
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            // Get image data from STB coordinates
            const uint8_t red = data[(y * width + x)*3 + 0];
            const uint8_t green = data[(y * width + x)*3 + 1];
            const uint8_t blue = data[(y * width + x)*3 + 2];

            // Get padded coordinate versions
            const int i = x + 1;
            const int j = height - 1 - y + 1; // (height - 1 - y) makes the direction consistent, add 1 to compensate for padding
            const size_t idx = (i * paddedSize.y) + (j);

            if (red == 0 && green == 0 && blue == 0) {
                simSnapshot.cell_type[idx] = CellType::Fluid;

                // TODO - connect to fluid.json params?
                if (constantVelocity) {
                    simSnapshot.velocity_x[idx] = 1;
                    simSnapshot.velocity_y[idx] = 0;
                } else {
                    if (occluded[y])
                        simSnapshot.velocity_x[idx] = 0;
                    else
                        simSnapshot.velocity_x[idx] = 1;
                    simSnapshot.velocity_y[idx] = 0;
                }
                if (interpolatePressure) {
                    // pressure = 1 at x=0
                    // pressure = 0 at x=width
                    // => pressure differential should push liquid out
                    simSnapshot.pressure[idx] = 1.0f - (x * 1.0f/width);
                } else {
                    // Constant pressure
                    simSnapshot.pressure[idx] = 1;
                }
            } else {
                simSnapshot.cell_type[idx] = CellType::Boundary;
                simSnapshot.velocity_x[idx] = 0;
                simSnapshot.velocity_y[idx] = 0;
                simSnapshot.pressure[idx] = 0;

                occluded[y] = 1;
            }
        }
    }

    // Set boundary squares
    for (uint32_t i = 0; i < paddedSize.x; i++) {
        // j = 0 => bottom row
        {
            uint32_t j = 0;
            const size_t idx = (i * paddedSize.y) + (j);
            simSnapshot.cell_type[idx] = CellType::Boundary;
        }
        // j = maximum => top row
        {
            uint32_t j = paddedSize.y - 1;
            const size_t idx = (i * paddedSize.y) + (j);
            simSnapshot.cell_type[idx] = CellType::Boundary;
        }
    }
    for (uint32_t j = 0; j < paddedSize.y; j++) {
        // i = 0 => left column
        {
            uint32_t i = 0;
            const size_t idx = (i * paddedSize.y) + (j);
            simSnapshot.cell_type[idx] = CellType::Boundary;
        }
        // i = maximum => right column
        {
            uint32_t i = paddedSize.x - 1;
            const size_t idx = (i * paddedSize.y) + (j);
            simSnapshot.cell_type[idx] = CellType::Boundary;
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
    subcommand->add_option("--constant-velocity", constantVelocity, "keep the velocity constant over the entire field")->default_val(true);
    subcommand->add_option("--interpolate-pressure", interpolatePressure, "interpolate pressure to create a gradient in the velocity direction")->default_val(true);
}
