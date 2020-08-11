//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "ISimFixedTimeRunner.h"

template<typename T>
class SimFixedTimeRunner : public ISimFixedTimeRunner {
public:
    ~SimFixedTimeRunner() override = default;

    SimSnapshot runForTime(const SimSnapshot& start, float timeToRun) override {
        auto sim = T(start);
        float currentTime = 0;
        while(currentTime < timeToRun) {
            currentTime += sim.tick();
            fprintf(stderr, "Current Time: %5g\r", currentTime);
        }
        fprintf(stderr, "\n");
        return sim.get_snapshot();
    }
};