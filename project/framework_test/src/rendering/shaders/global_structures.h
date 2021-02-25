//
// Created by samuel on 19/02/2021.
//

#include "util/glm.h"

namespace Shaders {
    using vec2 = glm::vec2;
    using vec4 = glm::vec4;
    using uint = uint32_t;
    using ivec2 = glm::ivec2;
    using ivec4 = glm::ivec4;
    using uvec2 = glm::uvec2;
    using uvec4 = glm::uvec4;

    // We don't need to preserve the "atomic" element of atomic_uint on the CPU side
    using atomic_uint = uint;

#include "global_structures.glsl"
}