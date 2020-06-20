//
// Created by samuel on 19/06/2020.
//

#pragma once

#include <cstddef>

/**
 * Generic base class for any sort of host-visible memory array.
 * Child classes must provide a constructor to allocate/map the memory, and a destructor to remove the memory.
 * Can be extended to support Vulkan mapped memory, CUDA unified memory.
 * IS NOT VIRTUAL - Don't try to do overrides etc.
 *  This is by design - all code that uses a HVA subclass should be aware of exactly the kind of memory it's holding.
 *  This is better for performance, avoiding any virtual call expense.
 * TODO: If this can refer/be used for CUDA kernel invocation, the name HostVisibleArray is sorta misleading as it's visible both on the host and device.
 *
 * @tparam T The type of elements present in the memory.
 */
 // TODO: This is doable as a subclass of std::unique_ptr with a custom deleter
template<typename T>
class HostVisibleArray {
public:
    // The pointer and element count cannot be modified during execution
    // T* const is a const pointer to T, instead of const T* which is a pointer to const T
    T* const pointer;
    size_t const elem_count;

    inline T* get() const {
        return pointer;
    }

    // begin() and end() iterators for compatibility with std:: algorithms
    inline T* begin() const {
        return pointer;
    }
    inline T* end() const {
        return pointer + elem_count;
    }

    // Allow move constructors
    HostVisibleArray(HostVisibleArray&&) noexcept = default;
    // Disallow copy constructors, as copying would create two references which would both try to delete the pointer
    HostVisibleArray(const HostVisibleArray<T>&) = delete;

protected:
    // The constructor is protected so it can only be called by subclasses,
    // which will override the HostVisibleArray with the correct behaviour.
    HostVisibleArray(T* pointer, size_t elem_count) : pointer(pointer), elem_count(elem_count) {}
    ~HostVisibleArray() = default;
};