//
// Created by samuel on 02/07/2020.
//

#include "CompareSubApp.h"

#include "validation/SimDumpDifferenceData.h"

void CompareSubApp::run() {
    LegacySimDump a = LegacySimDump::fromFile(fileA);
    LegacySimDump b = LegacySimDump::fromFile(fileB);

    auto diff = SimDumpDifferenceData(a, b);
    fprintf(stderr, "u:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", diff.u.errorMean, diff.u.errorVariance, diff.u.errorStdDev);
    fprintf(stderr, "v:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", diff.v.errorMean, diff.v.errorVariance, diff.v.errorStdDev);
    fprintf(stderr, "p:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", diff.p.errorMean, diff.p.errorVariance, diff.p.errorStdDev);
}

void CompareSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("a", fileA, "First file to compare")
            ->check(CLI::ExistingFile)
            ->required();

    subcommand->add_option("b", fileB, "Second file to compare")
            ->check(CLI::ExistingFile)
            ->required();
}
