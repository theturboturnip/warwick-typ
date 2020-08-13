//
// Created by samuel on 13/08/2020.
//

#include "simple.cuh"

__global__ void computeRHS_1per(const Array2D<const float> f, const Array2D<const float> g, const Array2D<const uint> is_fluid,
                                Array2D<float> rhs,
                                CommonParams p) {
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    // for i = [1, imax] inclusive
    // p.size.x = imax+2
    // so if i == imax + 1 == p.size.x - 1, ignore
    if (i == 0 || i >= p.size.x - 1) return;
    if (j == 0 || j >= p.size.y - 1) return;

    //if ((i >= p.size.x) || (j >= p.size.y)) return;

    if (!is_fluid.at(i, j)) return;

    float new_rhs = (
            (f.at(i, j)-f.at(i-1, j))/p.deltas.x +
            (g.at(i, j)-g.at(i, j-1))/p.deltas.y
    ) / p.timestep;
    rhs.at(i, j) = new_rhs;
}