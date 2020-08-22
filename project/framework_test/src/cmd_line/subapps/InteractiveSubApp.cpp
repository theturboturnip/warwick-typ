//
// Created by samuel on 22/08/2020.
//

#include "InteractiveSubApp.h"
#include <rendering/vulkan/VulkanWindow.h>
#include <simulation/file_format/FluidParams.h>
#include <simulation/file_format/SimSnapshot.h>

void InteractiveSubApp::run() {
    auto fluid_props = FluidParams::from_file(fluid_properties_file);
    auto initial = SimSnapshot::from_file(input_file);

    // Create window
    {
        auto appinfo = vk::ApplicationInfo(
                "Samuel Stark - Warwick Third-Year-Project",
                VK_MAKE_VERSION(1, 0, 0),
                "N/A",
                VK_MAKE_VERSION(1, 0, 0),
                VK_API_VERSION_1_1 // DCS only has v1.1, not v1.2
                );

        auto window = VulkanWindow(
                appinfo,
                { initial.simSize.pixel_size.x + 2, initial.simSize.pixel_size.y + 2 }
                        );

        window.main_loop();
    }
}

void InteractiveSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("fluid_properties", fluid_properties_file, "Fluid properties file")
            ->check(CLI::ExistingFile)
            ->required(true);
    subcommand->add_option("input_file", input_file, "Initial simulation state file")
            ->check(CLI::ExistingFile)
            ->required(true);
    converters.addBackendArgument(subcommand, backend);
}