//
// Created by samuel on 13/08/2020.
//

#pragma once

struct CommonParams {
    // Equivalent to imax+2, jmax+2
    ulong2 size;
    // Equivalent to (del_x, del_y)
    float2 deltas;

    float timestep;
};

template<typename T>
struct Array2D {
    T* pointer;
    size_t col_pitch_elems;

    __host__ Array2D(T* pointer, size_t col_pitch_elems)
        : pointer(pointer),
          col_pitch_elems(col_pitch_elems)
    {}
    /*__host__ Array2D(const Array2D<std::remove_const<T>>& copy)
        : pointer(copy.pointer),
          row_pitch_elems(copy.row_pitch_elems)
    {}*/
    __host__ static Array2D<T> from_pitch_bytes(T* pointer, size_t row_pitch_bytes) {
        return Array2D<T>(pointer, row_pitch_bytes / sizeof(T));
    }

    __host__ __device__ T& at(uint i, uint j) const {
        return pointer[i * col_pitch_elems + j];
    }
//    __host__ __device__ T at(uint i, uint j) const {
//        return pointer[j * row_pitch_elems + i];
//    }


    Array2D<const T> constify() const {
        return Array2D<const T>(pointer, col_pitch_elems);
    }
};