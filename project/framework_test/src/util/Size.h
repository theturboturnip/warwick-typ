//
// Created by samuel on 09/08/2020.
//

#pragma once

#include <utility>

template<typename T>
struct Size {
    T x, y;

    Size() : x(-1), y(-1) {}
    Size(T x, T y) : x(x), y(y) {}
    explicit Size(std::pair<T, T> pair) : x(pair.first), y(pair.second) {}
};