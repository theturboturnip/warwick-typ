//
// Created by samuel on 09/08/2020.
//

#pragma once

//#include <utility>

template<typename T>
struct Size {
    T x, y;

    Size() : x(-1), y(-1) {}
    Size(T x, T y) : x(x), y(y) {}
    // TODO - should we include a std::pair constructor?
    //Size(std::pair<T, T> pair) : x(pair.x), y(pair.y) {}
};