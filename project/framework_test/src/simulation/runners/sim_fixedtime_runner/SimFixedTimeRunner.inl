//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "ISimFixedTimeRunner.h"

template<typename T>
class SimFixedTimeRunner : public ISimFixedTimeRunner {
public:
    ~SimFixedTimeRunner() override = default;

    LegacySimDump runForTime(const LegacySimDump& start, float baseTimestep, float timeToRun) override {
        auto sim = T(start);
        float currentTime = 0;
        while(currentTime < timeToRun) {
            currentTime += sim.tick(baseTimestep);
        }
        return sim.dumpStateAsLegacy();
    }
};