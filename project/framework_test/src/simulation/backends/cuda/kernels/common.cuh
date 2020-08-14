//
// Created by samuel on 13/08/2020.
//

#pragma once

struct CommonParams {
    // Equivalent to imax+2, jmax+2
    ulong2 size;
    // Matrices are stored column-major - array[i][j] is adjacent to array[i][j+1]
    // The column pitch is stored separately, because it could be allocated to put the start of each one on a cache line boundary.
    ulong col_pitch_4byte;
    ulong col_pitch_redblack;

    // Equivalent to (del_x, del_y)
    float2 deltas;

    float timestep;

    __device__ inline uint flatten_4byte(uint i, uint j) const {
        // Arrays are column-contiguous
        return i * col_pitch_4byte + j;
    }
    __device__ inline bool in_real_range(uint i, uint j) const {
        // Keep i in range [1, imax] inclusive
        // size.x = imax+2 => reject if i==0, i >= imax+1 = size.x - 1
        // i,j can also go beyond those limits if p.size is not a multiple of 16 => use >= instead of ==
        if ((i == 0) || (i >= size.x - 1)) return false;
        if ((j == 0) || (j >= size.y - 1)) return false;
        return true;
    }
};

template<typename T>
struct Array2D {
    T* pointer;
    size_t col_pitch_elems;

    __host__ Array2D(T* pointer, size_t col_pitch_elems)
        : pointer(pointer),
          col_pitch_elems(col_pitch_elems)
    {}
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