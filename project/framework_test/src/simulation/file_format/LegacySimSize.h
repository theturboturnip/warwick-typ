//
// Created by samuel on 22/06/2020.
//

#pragma once

struct LegacySimSize {
    // Simulation block resolution
    uint32_t imax, jmax;
    inline size_t totalElements() const {
        return (imax+2) * (jmax+2);
    }
    // Simulation size in meters
    float xlength, ylength;
};