//
// Created by samuel on 12/02/2021.
//

#pragma once

#include <optional>
#include <utility>

#include "util/fatal_error.h"

/**
 * Class that is initialized with a value T, and can be moved without keeping a copy.
 *
 * By default, anything that can be copied-by-value will be copied in a trivial move constructor.
 * This means if you have a pointer in a class instance, which is then moved, the pointer will be in two places at once.
 * If that pointer is deleted in the destructor, it will be deleted twice.
 * This solves that issue - if it is moved, the previous instance will have the value removed, so you can't destruct it.
 * @tparam T
 */
template<class T>
class ForgetOnMove {
    std::optional<T> value;
public:
    // nullopt by default
    ForgetOnMove() : value(std::nullopt) {}
    // Implicit constructor so it can be assigned directly if needed.
    ForgetOnMove(T value) : value(value) {}
    // Move constructor sets the value of the `other` to std::nullopt.
    ForgetOnMove(ForgetOnMove&& o) noexcept : value(std::exchange(o.value, std::nullopt)) {}
    // No copy allowed.
    ForgetOnMove(const ForgetOnMove&) = delete;

    T& get() {
        //DASSERT(has_value())
        return value.value();
    }
    const T& get() const {
        //DASSERT(has_value())
        return value.value();
    }
    void set(T&& newValue) {
        FATAL_ERROR_UNLESS(value.empty(), "Can't set a value on top of an already-existing value");
        value = newValue;
    }
    bool has_value() const {
        return value.has_value();
    }
    // Explicitly disallow conversion to bool, as that might not be easy to understand
    operator bool() = delete;
    // Allow implicit T& conversion for ease of use
    operator const T& () const {
        return get();
    }
};