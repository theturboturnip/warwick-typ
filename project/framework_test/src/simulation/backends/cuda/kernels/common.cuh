//
// Created by samuel on 13/08/2020.
//

#pragma once

template<typename T>
using in_matrix = const T* const __restrict__;

// output matrix.
// The const in "T* const" is to denote that the pointer itself is constant, not that the data it points to is constant.
// i.e. it's a constant pointer to mutable data.
template<typename T>
using out_matrix = T* const __restrict__;

struct CommonParams {
    // Equivalent to imax+2, jmax+2
    uint2 size;
    // Equivalent to imax+2, (jmax+2)/2
    uint2 redblack_size;

    // Matrices are stored column-major - array[i][j] is adjacent to array[i][j+1]
    // The column pitch is stored separately, because it could be allocated to put the start of each one on a cache line boundary.
    uint col_pitch_4byte;
    uint col_pitch_redblack;

    // Equivalent to (del_x, del_y)
    float2 deltas;

    __device__ inline uint flatten_4byte(uint i, uint j) const {
        // Arrays are column-contiguous
        return i * col_pitch_4byte + j;
    }
    __device__ inline uint flatten_redblack(uint i, uint j) const {
        // Arrays are column-contiguous
        return i * col_pitch_redblack + j;
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