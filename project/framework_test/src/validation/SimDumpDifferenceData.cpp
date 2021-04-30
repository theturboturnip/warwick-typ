//
// Created by samuel on 23/06/2020.
//

#include <cmath>
#include "util/fatal_error.h"
#include "SimDumpDifferenceData.h"

SimDumpDifferenceData::SimDumpDifferenceData(const LegacySimDump &a, const LegacySimDump &b) :
    fluidmask(SimSnapshot::from_legacy(a).get_fluidmask()),
    u(fluidmask, a.u, b.u, false),
    v(fluidmask, a.v, b.v, false),
    p(fluidmask, a.p, b.p, false),
    p_meanadj(fluidmask, a.p, b.p, true)
    {
        FATAL_ERROR_UNLESS(fluidmask == SimSnapshot::from_legacy(b).get_fluidmask(), "A fluidmask doesn't match B fluidmask");
}

SimDumpDifferenceData::SimDumpDifferenceData(const SimSnapshot &a, const SimSnapshot &b) :
        fluidmask(a.get_fluidmask()),
        u(fluidmask, a.velocity_x, b.velocity_x, false),
        v(fluidmask, a.velocity_y, b.velocity_y, false),
        p(fluidmask, a.pressure, b.pressure, false),
        p_meanadj(fluidmask, a.pressure, b.pressure, true)
{
    FATAL_ERROR_UNLESS(fluidmask == b.get_fluidmask(), "A fluidmask doesn't match B fluidmask");
}

SingleDataDifference::SingleDataDifference(const std::vector<uint32_t>& fluid, const std::vector<float> &a, const std::vector<float> &b, bool subtractMean) : sqError() {
    FATAL_ERROR_UNLESS(a.size() == b.size(), "Size mismatch");
    FATAL_ERROR_UNLESS(a.size() == fluid.size(), "Size mismatch");

    const size_t N = a.size() - std::count(fluid.begin(), fluid.end(), 0);

    double mean_a = 0, mean_b = 0;
    // If we're supposed to subtract the mean, calculate the means here so the subtraction later works.
    // If we aren't supposed to subtract the mean, just set them to zero. The subtraction won't do anything.
    if (subtractMean) {
        for (size_t i = 0; i < a.size(); i++) {
            if (!fluid[i]) continue;
            mean_a += a[i];
            mean_b += b[i];
        }
        mean_a = mean_a / N;
        mean_b = mean_b / N;
    }

    double errorSum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!fluid[i]) continue;
        double error = (a[i] - mean_a) - (b[i] - mean_b);
        double error_sq = error * error;
        sqError.push_back(error_sq);
        errorSum += error_sq;
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
