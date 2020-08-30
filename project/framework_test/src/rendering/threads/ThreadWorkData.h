//
// Created by samuel on 26/08/2020.
//

#pragma once

#include <mutex>
#include <condition_variable>
#include <cstdint>

template<class T>
struct ThreadWorkData {
    // Put the mutex and condition variables in their own cacheline
    struct alignas(64) {
        std::mutex mutex;
        std::condition_variable readyForRead;
    } sync;

    // Put the data in a separate cacheline, so that i.e. someone reading the condition variable doesn't try to use the same cacheline as someone writing to the index.
    struct alignas(64) {
        int32_t index = -1;
        bool shouldJoin = false;
        T data;
    };
};