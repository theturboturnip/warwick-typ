//
// Created by samuel on 15/02/2021.
//

#pragma once

#include <optional>

struct SimAppProperties {
    bool useVsync;

    // If set, we are expected to lock the simulation frequency to this amount
    std::optional<int> lockSimFrequency;
    // If locking the frequency, don't tie it to real time.
    // If set to false, the simulation will only update N times a second in real-time.
    // If set to try, the simulation will update with a 1/N second sim-time tick every frame.
    bool unlockFrequencyFromRealTime;
    // Otherwise, the minimum frequency is set here - used if the simulation takes more than the simulated period to finish.
    // i.e. if a 33ms sim-time tick takes 100ms realtime (>33ms), don't try to simulate 100ms next time.
    int minUnlockedSimFrequency = 30;
};