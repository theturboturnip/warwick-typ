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

SingleDataDifference::SingleDataDifference(const std::vector<float> &a, const std::vector<float> &b) : error(a.size(), 0) {
    DASSERT(a.size() == b.size());

    const size_t N = a.size();

    for (size_t i = 0; i < N; i += BUCKET_SIZE) {
        buckets.push_back(ErrorBucket(i));
    }

    double errorSum = 0;
    for (size_t i = 0; i < N; i++) {
        error[i] = a[i] - b[i];
        errorSum += error[i];

        auto& bucket = buckets[i / BUCKET_SIZE];
        bucket.sumAbsError += fabs(error[i]);
        bucket.N++;
    }
    errorMean = errorSum / N;
    errorVariance = varianceOf(error, errorMean);
    errorStdDev = std::sqrt(errorVariance);

    // Generate the means for all buckets
    std::vector<double> bucketMeanErrors;
    double bucketMeanErrorMean = 0;
    for (auto& bucket : buckets) {
        bucket.meanAbsError = bucket.sumAbsError/N;
        bucketMeanErrorMean += bucket.meanAbsError;
        bucketMeanErrors.push_back(bucket.meanAbsError);
    }
    bucketMeanErrorMean /= buckets.size();
    bucketMeanErrorVariance = varianceOf(bucketMeanErrors, bucketMeanErrorMean);
    bucketMeanErrorStdDev = std::sqrt(bucketMeanErrorVariance);

    // Sort the buckets by meanError
    std::sort(buckets.begin(), buckets.end(), [](const ErrorBucket& b1, const ErrorBucket& b2){
        return b1.meanAbsError > b2.meanAbsError;
    });
}
double SingleDataDifference::varianceOf(std::vector<double> values, double mean) {
    double sumDiffSquared = 0;
    for (const auto e : error) {
        double diff = e - errorMean;
        sumDiffSquared += (diff * diff);
    }
    return sumDiffSquared/(values.size()-1);
}
void SingleDataDifference::print_details() {
    printf("\tMean:\t\t\t%g\n\tStd. Dev:\t\t%g\n\tBucket Std. Dev:\t%g\nBuckets by Mean Error\n", errorMean, errorStdDev, bucketMeanErrorStdDev);
    for (size_t b_idx = 0; b_idx < 5 && b_idx < buckets.size(); b_idx++) {
        const auto& b = buckets[b_idx];
        printf("\t\t#%lu - [%5lu]\t%g\n", b_idx+1, b.i, b.meanAbsError);
    }
}
