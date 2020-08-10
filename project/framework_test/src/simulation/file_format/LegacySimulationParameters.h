//
// Created by samuel on 22/06/2020.
//

#pragma once

struct LegacySimulationParameters {
    // Simulation block resolution
    int imax, jmax;
    inline int totalElements() const {
        return (imax+2) * (jmax+2);
    }
    // Simulation size in meters
    float xlength, ylength;
};