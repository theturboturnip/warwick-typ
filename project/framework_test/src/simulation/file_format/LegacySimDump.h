//
// Created by samuel on 20/06/2020.
//

#pragma once

#include <vector>
#include <string>
#include "simulation/SimulationParameters.h"

struct LegacySimDump {
    SimulationParameters params;

    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> p;

    std::vector<char> flag;

    static LegacySimDump fromFile(std::string path);
    void saveToFile(std::string path);

    std::string debugString();
};