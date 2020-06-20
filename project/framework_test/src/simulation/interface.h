//
// Created by samuel on 20/06/2020.
//

#pragma once

struct LegacySimDump;

class ISimulation {
protected:
    ISimulation() = default;

    bool m_started = false;
    float m_currentTime = 0;

public:
    virtual ~ISimulation() = default;

    inline float started() const {
        return m_started;
    }

    inline float currentTime() const {
        return m_currentTime;
    }

    virtual void tick(float expectedTimestep) = 0;

    virtual void loadFromLegacy(const LegacySimDump& dump) = 0;
    virtual LegacySimDump dumpStateAsLegacy() = 0;
};