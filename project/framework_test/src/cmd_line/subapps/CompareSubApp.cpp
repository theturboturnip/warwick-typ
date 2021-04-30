//
// Created by samuel on 02/07/2020.
//

#include "CompareSubApp.h"

#include "validation/SimDumpDifferenceData.h"

void CompareSubApp::run() {
    auto a = SimSnapshot::from_file(fileA);
    auto b = SimSnapshot::from_file(fileB);

    auto diff = SimDumpDifferenceData(a, b);
    printf("Velocity X:\n");
    diff.u.print_details();
    printf("Velocity Y:\n");
    diff.v.print_details();
    printf("Pressure:\n");
    diff.p.print_details();
    printf("Pressure (Mean Adjusted):\n");
    diff.p_meanadj.print_details();
    //fprintf(stderr, "u:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", diff.u.errorMean, diff.u.errorVariance, diff.u.errorStdDev);
    //fprintf(stderr, "v:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", diff.v.errorMean, diff.v.errorVariance, diff.v.errorStdDev);
    //fprintf(stderr, "p:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", diff.p.errorMean, diff.p.errorVariance, diff.p.errorStdDev);
}

void CompareSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("a", fileA, "First file to compare")
            ->check(CLI::ExistingFile)
            ->required();

    subcommand->add_option("b", fileB, "Second file to compare")
            ->check(CLI::ExistingFile)
            ->required();
}
