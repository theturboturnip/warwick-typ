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

SingleDataDifference::SingleDataDifference(const std::vector<float> &a, const std::vector<float> &b) : error(a.size(), 0) {
    DASSERT(a.size() == b.size());

    double errorSum = 0;
    const int N = a.size();
    for (int i = 0; i < N; i++) {
        error[i] = a[i] - b[i];
        errorSum += error[i];
    }

    errorMean = errorSum / N;

    double sumDiffSquared = 0;
    for (const auto e : error) {
        double diff = e - errorMean;
        sumDiffSquared += (diff * diff);
    }
    errorVariance = sumDiffSquared/(N-1);
    errorStdDev = std::sqrt(errorVariance);
}