//
// Created by samuel on 08/02/2021.
//

#pragma once

#include <algorithm>
#include <vector>

#include "fatal_error.h"

template<class T, class TList>
static T selectIfPossible(TList list, std::vector<T> wishes) {
    for (const auto& wish : wishes) {
        if (std::find(list.begin(), list.end(), wish) != list.end())
            return wish;
    }

    FATAL_ERROR("selectIfPossible failed to select a %s, no wish was present.\n", typeid(T).name());
}

template<class T, class TList>
static T selectOrFallback(TList list, std::vector<T> wishes) {
    for (const auto& wish : wishes) {
        if (std::find(list.begin(), list.end(), wish) != list.end())
            return wish;
    }

    // Return the first one
    if (list.begin() == list.end())
        FATAL_ERROR("selectOrFallback failed to select a %s, list is empty\n", typeid(T).name());
    return *list.begin();
}

template<class T, class TList, class Selector>
static T selectAnySuitable(TList list, Selector selector) {
    for (const auto& item : list) {
        if (selector(item))
            return item;
    }

    FATAL_ERROR("selectAnySuitable failed to select a %s, no options were correct.\n", typeid(T).name());
}