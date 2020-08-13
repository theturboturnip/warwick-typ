//
// Created by samuel on 20/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimulationParameters.h"
#include <string>
#include <vector>

// Definitions of flags in the flag vector
// TODO: Refactor into namespace?
#define C_B      0x0000   /* This cell is an obstacle/boundary cell */
#define B_N      0x0001   /* This obstacle cell has a fluid cell to the north */
#define B_S      0x0002   /* This obstacle cell has a fluid cell to the south */
#define B_W      0x0004   /* This obstacle cell has a fluid cell to the west */
#define B_E      0x0008   /* This obstacle cell has a fluid cell to the east */
#define B_NW     (B_N | B_W)
#define B_SW     (B_S | B_W)
#define B_NE     (B_N | B_E)
#define B_SE     (B_S | B_E)
#define B_NSEW   (B_N | B_S | B_E | B_W)

#define C_F      0x0010   /* This cell is a fluid cell */

struct LegacySimDump {
    LegacySimDump() = default;
    explicit LegacySimDump(LegacySimulationParameters params);

    LegacySimulationParameters params;

    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> p;

    std::vector<char> flag;

    static LegacySimDump fromFile(std::string path);
    void saveToFile(std::string path);

    std::string debugString();
};