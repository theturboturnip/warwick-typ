//
// Created by samuel on 15/02/2021.
//

#pragma once

#include <optional>

struct SimAppProperties {
    bool useVsync;

    // If set, we are expected to lock the simulation frequency to this amount
    std::optional<int> fixedSimFrequency;
    // Otherwise, the minimum frequency is set here - used if the simulation takes more than the simulated period to finish.
    // i.e. if a 33ms sim-time tick takes 100ms realtime (>33ms), don't try to simulate 100ms next time.
    int minUnlockedSimFrequency = 30;
    // Maximum unlocked frequency - Just a sensible limit i.e. don't run faster than 250Hz because small floating-point inaccuracy
    int maxUnlockedSimFrequency = 250;

    // Independent of fixing the frequency.
    // If set to true, the simulation will only update N times a second in real-time.
    // If set to false, the simulation will update with a 1/N second sim-time tick every frame.
    bool matchFrequencyToRealTime;

    // Maximum amount of particles that could be rendered
    uint32_t maxParticles = 100000;
    uint32_t maxParticlesEmittedPerFrame = 16;
    uint32_t maxParicleEmitters = 16;
    uint32_t maxVectorArrows = 10000;
};