//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "ISim10sRunner.h"

template<typename T>
class Sim10sRunner : public ISim10sRunner {
public:
    ~Sim10sRunner() override = default;

    LegacySimDump runFor10s(const LegacySimDump& start, float baseTimestep) override {
        auto sim = T(start);
        float currentTime = 0;
        while(currentTime < 10) {
            currentTime += sim.tick(baseTimestep);
        }
        return sim.dumpStateAsLegacy();
    }
};