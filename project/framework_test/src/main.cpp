#include <CLI/CLI11.hpp>
#include <chrono>
#include <cmd_line/CommandLineParser.h>
#include <vulkan/vulkan.hpp>

#include "simulation/file_format/LegacySimDump.h"
#include "validation/SimDumpDifferenceData.h"

void printSimDumpDifference(const SimDumpDifferenceData& data) {
    fprintf(stderr, "u:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.u.errorMean, data.u.errorVariance, data.u.errorStdDev);
    fprintf(stderr, "v:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.v.errorMean, data.v.errorVariance, data.v.errorStdDev);
    fprintf(stderr, "p:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.p.errorMean, data.p.errorVariance, data.p.errorStdDev);
}

int main(int argc, const char* argv[]) {
    return CommandLineParser().parseArguments(argc, argv);
}