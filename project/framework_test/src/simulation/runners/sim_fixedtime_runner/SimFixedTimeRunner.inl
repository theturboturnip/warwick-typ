//
// Created by samuel on 30/06/2020.
//

#pragma once

#include "ISimFixedTimeRunner.h"

#include "memory/FrameSetAllocator.h"

template<typename T, MType MemType>
class SimFixedTimeRunner : public ISimFixedTimeRunner {
    int ticks = 0;
public:
    ~SimFixedTimeRunner() override = default;

    SimSnapshot runForTime(const FluidParams& simParams, const SimSnapshot& start, float timeToRun, float forcedMaxTimestep) override {
        const size_t frameCount = 1;
        FrameSetAllocator<MemType, typename T::Frame> allocator(start, frameCount);
        auto sim = T(allocator.frames, simParams, start);
        float currentTime = 0;
        ticks = 0;
        while(currentTime < timeToRun) {
            float maxTimestep = sim.findMaxTimestep();
            if (forcedMaxTimestep > 0 && maxTimestep > forcedMaxTimestep)
                maxTimestep = forcedMaxTimestep;
            if (currentTime + maxTimestep > timeToRun)
                maxTimestep = timeToRun - currentTime;
//            fprintf(stdout, "t: %f\tts: %f\r", currentTime, maxTimestep);
            sim.tick(maxTimestep, 0);
            currentTime += maxTimestep;
            ticks++;
        }
//        fprintf(stdout, "\n");
        return sim.get_snapshot();
    }

    int requiredTicks() override {
        return ticks;
    }
};