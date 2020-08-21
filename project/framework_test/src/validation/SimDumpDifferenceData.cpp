//
// Created by samuel on 23/06/2020.
//

#include <cmath>
#include "util/fatal_error.h"
#include "SimDumpDifferenceData.h"

SimDumpDifferenceData::SimDumpDifferenceData(const LegacySimDump &a, const LegacySimDump &b) :
    u(a.u, b.u),
    v(a.v, b.v),
    p(a.p, b.p)
    {}

SimDumpDifferenceData::SimDumpDifferenceData(const SimSnapshot &a, const SimSnapshot &b) :
        u(a.velocity_x, b.velocity_x),
        v(a.velocity_y, b.velocity_y),
        p(a.pressure, b.pressure)
{}

SingleDataDifference::SingleDataDifference(const std::vector<float> &a, const std::vector<float> &b) : sqError(a.size(), 0) {
    DASSERT(a.size() == b.size());

    const size_t N = a.size();

    double errorSum = 0;
    for (size_t i = 0; i < N; i++) {
        double error = a[i] - b[i];
        sqError[i] = error * error;
        errorSum += sqError[i];
    }
    sqErrorMean = errorSum / N;
    sqErrorVariance = varianceOf(sqError, sqErrorMean);
    sqErrorStdDev = std::sqrt(sqErrorVariance);

    isAccurate = sqErrorMean < MeanSquareErrorHeuristic;
    isPrecise = sqErrorStdDev < StdDevSquareErrorHeuristic;
}
double SingleDataDifference::varianceOf(std::vector<double> values, double mean) {
    double sumDiffSquared = 0;
    for (const auto e : values) {
        double diff = e - mean;
        sumDiffSquared += (diff * diff);
    }
    return sumDiffSquared/(values.size()-1);
}
void SingleDataDifference::print_details() {
#define GREEN_TEXT "\e[92;1m"
#define RED_TEXT "\e[91;1m"
#define CLEAR_TEXT "\e[0m"

    printf("\tSq. Error Mean:\t\t%11g\t%s\n",
           sqErrorMean,
           (isAccurate ? (GREEN_TEXT "ACCURATE" CLEAR_TEXT) : (RED_TEXT "INACCURATE" CLEAR_TEXT))
           );
    printf("\tSq. Error Std. Dev:\t%11g\t%s\n",
           sqErrorStdDev,
           (isPrecise ? (GREEN_TEXT "PRECISE" CLEAR_TEXT) : (RED_TEXT "IMPRECISE" CLEAR_TEXT))
           );

#undef GREEN_TEXT
#undef RED_TEXT
#undef CLEAR_TEXT
}
