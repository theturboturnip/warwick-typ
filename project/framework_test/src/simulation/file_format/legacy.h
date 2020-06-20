//
// Created by samuel on 20/06/2020.
//

#pragma once

#include <vector>
#include <string>

struct LegacySimDump {
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> p;

    std::vector<char> flag;

    // Simulation block resolution
    int imax, jmax;
    inline int totalElements() const {
        return (imax+2) * (jmax+2);
    }
    // Simulation size in meters
    float xlength, ylength;

    static LegacySimDump fromFile(std::string path);
    void saveToFile(std::string path);

    std::string debugString();
};