//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "ISimFixedTimeRunner.h"

template<typename T, typename AllocType>
class SimFixedTimeRunner : public ISimFixedTimeRunner {
    std::unique_ptr<AllocType> alloc;

public:
    SimFixedTimeRunner() : alloc(std::make_unique<AllocType>()) {}
    ~SimFixedTimeRunner() override = default;

    SimSnapshot runForTime(const FluidParams & simParams, const SimSnapshot& start, float timeToRun) override {
        auto sim = T(alloc.get(), simParams, start);
        float currentTime = 0;
        while(currentTime < timeToRun) {
            float maxTimestep = sim.findMaxTimestep();
            if (currentTime + maxTimestep > timeToRun)
                maxTimestep = timeToRun - currentTime;
            fprintf(stdout, "t: %f\tts: %f\r", currentTime, maxTimestep);
            sim.tick(maxTimestep);
            currentTime += maxTimestep;
        }
        fprintf(stdout, "\n");
        return sim.get_snapshot();
    }
};