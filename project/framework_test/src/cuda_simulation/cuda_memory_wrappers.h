//
// Created by samuel on 18/06/2020.
//

#pragma once

#include <cstdlib>
#include <vector>
#include <cstring>

/**
 * Managed pointer for CUDA allocated arrays of T.
 * Allocates a CUDA Unified Memory array of T on construction, frees the array when destroyed.
 * Essentially std::unique_ptr for CUDA arrays.
 * Exists for T = (float, double, uint32)
 *
 * @tparam T The type of elements in the array.
 */
template<typename T>
class CUDAUnified1DArray {
public:
    explicit CUDAUnified1DArray(size_t elemCount);
    ~CUDAUnified1DArray();

    // Delete the copy constructor to prevent accidental data copying using "x = y;"
    CUDAUnified1DArray(const CUDAUnified1DArray&) = delete;
    // Allow the array to be std::move-d if, for example, it needs to be returned from a function.
    CUDAUnified1DArray(CUDAUnified1DArray&&) noexcept = default;

    const size_t elemCount;

    explicit operator T*() {
        return pointer;
    }
    T* get() {
        return pointer;
    }

    // begin() and end() to allow for std:: algorithms and fromContainer to work.
    // Enable only if necessary, as that's really a bad idea.
    /*
    T* begin() {
        return pointer;
    }
    T* end() {
        return pointer + elemCount;
    }
     */

    // Use an arbitrary Iter type to allow any Iterator to be used
    // This means iterators from std::vector, std::list, or just raw pointers could be used.
    template<typename Iter>
    static CUDAUnified1DArray<T> fromContainer(Iter it, Iter end) {
        auto array = CUDAUnified1DArray<T>(end - it);
        // Get the destination pointer
        T* data = array.get();
        // For every element between it and end,
        for (; it != end; ++it) {
            // Copy it into the destination
            *data = *it;
            data++;
        }
        return std::move(array);
    }

protected:
    T* pointer;
};

/**
 * Managed set of CUDA Unified Arrays.
 * Arrays can be allocated or moved in on construction.
 * The arrays are freed when the Unified2DArray is destroyed.
 * Exists for T = (float, double, uint32)
 *
 * @tparam T The type of elements in both arrays.
 */
template<typename T>
class CUDAUnified2DArray {
public:
    // Element Move Constructor - each array is std::move-d in from another source
    CUDAUnified2DArray(CUDAUnified1DArray<T>&& x, CUDAUnified1DArray<T>&& y) : x(x), y(y) {}
    // Element Allocation Constructor - each array is allocated
    CUDAUnified2DArray(size_t elemCountX, size_t elemCountY) : x(elemCountX), y(elemCountY) {}

    // Disable copying the array pair, because each array cannot be copied.
    CUDAUnified2DArray(const CUDAUnified2DArray&) = delete;
    // Allow the array pair to be std::move-d if, for example, it needs to be returned from a function.
    CUDAUnified2DArray(CUDAUnified2DArray&&) noexcept = default;

    const CUDAUnified1DArray<T> x, y;
};
