//
// Created by samuel on 02/07/2020.
//

#pragma once

#include "subapps/ISubApp.h"

#include <memory>

class CommandLineParser {
public:
    int parseArguments(int argc, const char* argv[]);

    constexpr static const char* APP_NAME = "2D Fluid Simulator";
};