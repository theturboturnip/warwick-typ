#include <vulkan/vulkan.hpp>
#include <simulation/file_format/legacy.h>

#include "simulation/get_sims.h"

SimulationBackend selectBackend() {
#if false && CUDA_ENABLED
    return SimulationBackend::CUDA;
#endif

    return SimulationBackend::CpuSimple;
    //return SimulationBackend::Null;
}

int main() {
    auto backend = selectBackend();
    std::unique_ptr<ISimulation> sim = getSimulation(backend);

    ISimulation& actualSim = *sim;
    fprintf(stderr, "acquired simulation %s\n", typeid(actualSim).name());

    LegacySimDump dump = LegacySimDump::fromFile("initial.bin");
    sim->loadFromLegacy(dump);
    fprintf(stderr, "startdump: %s", dump.debugString().c_str());

    const float targetTime = 10.0f;
    const float timestep = 0.01f;//1.0/120.0;
    while(sim->currentTime() < targetTime) {
        fprintf(stderr, "\rb: %5g", sim->currentTime());
        sim->tick(timestep);
        //fprintf(stderr, "\re: %5g", sim->currentTime());
    }

    LegacySimDump endDump = sim->dumpStateAsLegacy();
    fprintf(stderr, "enddump: %s", endDump.debugString().c_str());
    endDump.saveToFile("output.bin");

    return 0;
}