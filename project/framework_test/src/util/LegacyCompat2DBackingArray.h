//
// Created by samuel on 22/06/2020.
//

#pragma once

template<typename T>
class LegacyCompat2DBackingArray {
    std::vector<T> m_backing;
    std::vector<T*> m_as2D;

public:
    LegacyCompat2DBackingArray(int width, int height, T default_back) : m_backing(width*height, default_back), m_as2D(width, nullptr) {
        for (int x = 0; x < width; x++) {
            m_as2D[x] = &m_backing[x * height];
        }
    }

    LegacyCompat2DBackingArray(const std::vector<T>& backing, int width, int height) : m_backing(backing), m_as2D(width, nullptr) {
        for (int x = 0; x < width; x++) {
            m_as2D[x] = &m_backing[x * height];
        }
    }

    /*operator std::vector<T*>&() {
        return m_as2D;
    }*/
    T* operator[](int index) {
        return m_as2D[index];
    }

    const std::vector<T>& getBacking() {
        return m_backing;
    }
};