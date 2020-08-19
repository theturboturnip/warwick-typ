//
// Created by samuel on 23/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"
#include <vector>

// Heuristic for determining if mean of square error is OK or not
constexpr double MeanSquareErrorHeuristic = 1E-14; // Mean-magnitude-of-error should be < 1E-7, so Mean-square-error should be < 1E-14
// Heuristic for determining if std. deviation of square error is OK or not
// For a normal distribution, 95% of values are within two standard deviations of the mean. This is one on either side.
// if stddev >= mean, then many values are 0. *NOTE* this doesn't calculate the std.dev relative to the actual mean, just takes the MeanSquareErrorHeuristic as the "assumed mean".
constexpr double StdDevSquareErrorHeuristic = 1E-14;

struct SingleDataDifference {
    std::vector<double> sqError;
    double sqErrorMean;
    double sqErrorVariance;
    double sqErrorStdDev;

    bool isAccurate;
    bool isPrecise;

    SingleDataDifference(const std::vector<float>& a, const std::vector<float>& b);
    double varianceOf(std::vector<double> values, double mean);
    void print_details();
};

struct SimDumpDifferenceData {
    SingleDataDifference u, v, p;

    SimDumpDifferenceData(const LegacySimDump& a, const LegacySimDump& b);
    SimDumpDifferenceData(const SimSnapshot& a, const SimSnapshot& b);
};