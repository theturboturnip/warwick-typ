#include <vulkan/vulkan.hpp>

#include "simulation/get_sims.h"
#include "simulation/file_format/legacy.h"
#include "validation/SimDumpDifferenceData.h"

void printSimDumpDifference(const SimDumpDifferenceData& data) {
    fprintf(stderr, "u:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.u.errorMean, data.u.errorVariance, data.u.errorStdDev);
    fprintf(stderr, "v:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.v.errorMean, data.v.errorVariance, data.v.errorStdDev);
    fprintf(stderr, "p:\n\tmean: \t%g\n\tvariance: \t%g\n\tstddev: \t%g\n", data.p.errorMean, data.p.errorVariance, data.p.errorStdDev);
}

SimulationBackend selectBackend() {
#if false && CUDA_ENABLED
    return SimulationBackend::CUDA;
#endif

    return SimulationBackend::CpuSimple;
    //return SimulationBackend::Null;
}

int main() {
    auto backend = selectBackend();
    std::unique_ptr<ISimTickedRunner> sim = getSimulation(backend);

    ISimTickedRunner& actualSim = *sim;
    fprintf(stderr, "acquired simulation %s\n", typeid(actualSim).name());

    LegacySimDump dump = LegacySimDump::fromFile("initial.bin");
    sim->loadFromLegacy(dump);
    fprintf(stderr, "startdump: %s", dump.debugString().c_str());

    auto zeroDiff = SimDumpDifferenceData(dump, dump);
    fprintf(stderr, "Expected Zero Diff\n");
    printSimDumpDifference(zeroDiff);

    const float targetTime = 10.0f;
    const float timestep = 0.01f;//1.0/120.0;
    while(sim->currentTime() < targetTime) {
        float realstep = sim->tick(timestep);
        fprintf(stderr, "\rt: %5g dt: %5g", sim->currentTime(), realstep);
        //fprintf(stderr, "\re: %5g", sim->currentTime());
    }
    fprintf(stderr, "\n");

    LegacySimDump endDump = sim->dumpStateAsLegacy();
    fprintf(stderr, "enddump: %s", endDump.debugString().c_str());
    endDump.saveToFile("output.bin");

    LegacySimDump targetEndDump = LegacySimDump::fromFile("target.bin");

    auto outputTargetDiff = SimDumpDifferenceData(endDump, targetEndDump);
    fprintf(stderr, "Output->Target\n");
    printSimDumpDifference(outputTargetDiff);

    return 0;
}