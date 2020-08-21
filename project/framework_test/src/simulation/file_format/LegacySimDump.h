//
// Created by samuel on 20/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimSize.h"
#include <string>
#include <vector>

#include "simulation/backends/original/constants.h"

struct LegacySimDump {
    LegacySimDump() = default;
    explicit LegacySimDump(LegacySimSize params);

    LegacySimSize simSize;

    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> p;

    std::vector<char> flag;

    static LegacySimDump fromFile(std::string path);
    void saveToFile(std::string path);

    std::string debugString();
};