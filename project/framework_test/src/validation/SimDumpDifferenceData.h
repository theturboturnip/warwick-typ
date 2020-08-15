//
// Created by samuel on 23/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"
#include <vector>

struct SingleDataDifference {
    std::vector<float> error;
    double errorMean;
    double errorAbsMean;
    double errorVariance;
    double errorStdDev;

    SingleDataDifference(const std::vector<float>& a, const std::vector<float>& b);
};

struct SimDumpDifferenceData {
    SingleDataDifference u, v, p;

    SimDumpDifferenceData(const LegacySimDump& a, const LegacySimDump& b);
    SimDumpDifferenceData(const SimSnapshot& a, const SimSnapshot& b);
};