//
// Created by samuel on 25/02/2021.
//

#pragma once

#include <array>

enum class ScalarQuantity : uint32_t {
    None=0,
    VelocityX=1,
    VelocityY=2,
    VelocityMagnitude=3,
    Pressure=4,
    Vorticity=5
};
extern std::array<const char*, 6> scalarQuantityStrs;
enum class VectorQuantity : uint32_t {
    None=0,
    Velocity=1
};
extern std::array<const char*, 2> vectorQuantityStrs;

enum class ParticleTrailType : uint32_t {
    None=0,
    Streakline=1,
    Pathline=2, // < are these different?
    Ribbon=3    // <
};
extern std::array<const char*, 4> particleTrailTypeStrs;