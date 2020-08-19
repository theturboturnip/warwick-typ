//
// Created by samuel on 23/06/2020.
//

#pragma once

#include "simulation/file_format/LegacySimDump.h"
#include "simulation/file_format/SimSnapshot.h"
#include <vector>

constexpr size_t BUCKET_SIZE = 16;

struct ErrorBucket {
    ErrorBucket(size_t i) : i(i) {}

    size_t i;

    double sumAbsError = 0;
    int N = 0;

    double meanAbsError = -1;
};

struct SingleDataDifference {
    std::vector<ErrorBucket> buckets;
    std::vector<double> error;
    double errorMean;
    double errorVariance;
    double errorStdDev;
    double bucketMeanErrorVariance;
    double bucketMeanErrorStdDev;

    SingleDataDifference(const std::vector<float>& a, const std::vector<float>& b);
    double varianceOf(std::vector<double> values, double mean);
    void print_details();
};

struct SimDumpDifferenceData {
    SingleDataDifference u, v, p;

    SimDumpDifferenceData(const LegacySimDump& a, const LegacySimDump& b);
    SimDumpDifferenceData(const SimSnapshot& a, const SimSnapshot& b);
};