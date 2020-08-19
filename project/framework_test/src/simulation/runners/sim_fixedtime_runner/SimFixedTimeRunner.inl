//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "ISimFixedTimeRunner.h"

template<typename T>
class SimFixedTimeRunner : public ISimFixedTimeRunner {
public:
    ~SimFixedTimeRunner() override = default;

    SimSnapshot runForTime(const SimParams& simParams, const SimSnapshot& start, float timeToRun) override {
        auto sim = T(simParams, start);
        float currentTime = 0;
        while(currentTime < timeToRun) {
            float maxTimestep = sim.findMaxTimestep();
            if (currentTime + maxTimestep > timeToRun)
                maxTimestep = timeToRun - currentTime;
            //fprintf(stdout, "t: %f\tts: %f\n", currentTime, maxTimestep);
            sim.tick(maxTimestep);
            currentTime += maxTimestep;
        }
        //fprintf(stderr, "\n");
        return sim.get_snapshot();
    }
};