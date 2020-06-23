//
// Created by samuel on 23/06/2020.
//

#pragma once

#include <vector>
#include "simulation/file_format/legacy.h"

struct SingleDataDifference {
    std::vector<float> error;
    double errorMean;
    double errorVariance;
    double errorStdDev;

    SingleDataDifference(const std::vector<float>& a, const std::vector<float>& b);
};

struct SimDumpDifferenceData {
    SingleDataDifference u, v, p;

    SimDumpDifferenceData(const LegacySimDump& a, const LegacySimDump& b);
};