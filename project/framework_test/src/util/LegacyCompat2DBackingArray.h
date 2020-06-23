//
// Created by samuel on 22/06/2020.
//

#pragma once

#include "fatal_error.h"
#include <vector>
#include <gsl/span>

template<typename T>
class LegacyCompat2DBackingArray {
    std::vector<T> m_backing;
    std::vector<gsl::span<T>> m_as2D;

public:
    LegacyCompat2DBackingArray(int width, int height, T default_back) : m_backing(width*height, default_back), m_as2D(width) {
        build2D(width, height);
    }

    LegacyCompat2DBackingArray(const std::vector<T>& backing, int width, int height) : m_backing(backing), m_as2D(width) {
        build2D(width, height);
    }

    gsl::span<T> operator[](int index) {
        return m_as2D[index];
    }

    const std::vector<T>& getBacking() {
        return m_backing;
    }

private:
    void build2D(int width, int height) {
        DASSERT(m_backing.size() == width*height);
        for (int x = 0; x < width; x++) {
            m_as2D[x] = gsl::span<T>(&m_backing[x * height], height);
        }
    }
};