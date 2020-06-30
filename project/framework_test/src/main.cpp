#include <vulkan/vulkan.hpp>
#include <chrono>

#include "simulation/SimulationBackendEnum.h"
#include "simulation/file_format/LegacySimDump.h"
#include "simulation/runners/sim_10s_runner/ISim10sRunner.h"
#include "simulation/runners/sim_ticked_runner/ISimTickedRunner.h"
#include "validation/SimDumpDifferenceData.h"

void printSimDumpDifference(const SimDumpDifferenceData& data) {
    fprintf(stderr, "u:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.u.errorMean, data.u.errorVariance, data.u.errorStdDev);
    fprintf(stderr, "v:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.v.errorMean, data.v.errorVariance, data.v.errorStdDev);
    fprintf(stderr, "p:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.p.errorMean, data.p.errorVariance, data.p.errorStdDev);
}

SimulationBackendEnum selectBackend() {
#if false && CUDA_ENABLED
    return SimulationBackendEnum::CUDA;
#endif

    return SimulationBackendEnum::CpuOptimized;
    //return SimulationBackendEnum::CpuSimple;
    //return SimulationBackendEnum::Null;
}

struct SimRunData {
    LegacySimDump finalDump;
    double timeInSeconds = 0;
};

SimRunData runSimFor10sWithTicked(SimulationBackendEnum backend, const LegacySimDump& initial) {
    auto sim = ISimTickedRunner::getForBackend(backend);

    /*
    ISimTickedRunner& actualSim = *sim;
    fprintf(stderr, "acquired simulation %s\n", typeid(actualSim).name());
    */

    sim->loadFromLegacy(initial);

    auto start = std::chrono::steady_clock::now();

    const float targetTime = 10.0f;
    const float baseTimestep = 1.0f/30.0f;
    while(sim->currentTime() < targetTime) {
        float realstep = sim->tick(baseTimestep);
        fprintf(stderr, "\rt: %5g dt: %5g", sim->currentTime(), realstep);
        //fprintf(stderr, "\re: %5g", sim->currentTime());
    }
    fprintf(stderr, "\n");

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;

    return SimRunData{ .finalDump = sim->dumpStateAsLegacy(), .timeInSeconds = diff.count() };
}

SimRunData runSimFor10sWith10sRunner(SimulationBackendEnum backend, const LegacySimDump& initial) {
    auto sim = ISim10sRunner::getForBackend(backend);

    auto start = std::chrono::steady_clock::now();

    const float baseTimestep = 1.0f/30.0f;
    LegacySimDump output = sim->runFor10s(initial, baseTimestep);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;

    return SimRunData{ .finalDump = std::move(output), .timeInSeconds = diff.count() };
}

int main() {
    auto backend = selectBackend();

    LegacySimDump initial = LegacySimDump::fromFile("initial.bin");

    SimRunData runData = runSimFor10sWith10sRunner(backend, initial);

    LegacySimDump& output = runData.finalDump;
    fprintf(stderr, "performed calc in %f seconds\n", runData.timeInSeconds);
    fprintf(stderr, "enddump: %s", output.debugString().c_str());
    output.saveToFile("output.bin");

    LegacySimDump targetEndDump = LegacySimDump::fromFile("target.bin");
    auto outputTargetDiff = SimDumpDifferenceData(output, targetEndDump);
    fprintf(stderr, "Output->Target\n");
    printSimDumpDifference(outputTargetDiff);

    return 0;
}