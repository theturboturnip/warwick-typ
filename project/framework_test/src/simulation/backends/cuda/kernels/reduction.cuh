//
// Created by samuel on 17/08/2020.
//

#pragma once

#include "common.cuh"

/**
 * Single dimensional kernel that performs a reduction over input data.
 * Output array must be at least of length <gridSize.x>.
 * Output array is filled with [Func(Preproc(input[0:blockDim]...)), Func(Preproc(input[blockDim:2*blockDim]...)) etc.]
 * Expects blockDim.x to be a power of 2. For the __syncthreads to be worthwhile, the blockSize should be >32, otherwise each block is a single warp.
 * Allocates sizeof(float) * blockDim.x of shared memory.
 * input and output *MAY NOT* alias
 *
 * Based on https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *
 * @tparam Preproc
 * @tparam Func
 * @param input
 * @param output
 * @param input_length
 */
template<typename Preproc, typename Func>
__global__ void reduce_simple(in_matrix<float> input, size_t input_length, out_matrix<float> output,
                              Preproc preprocess, Func f) {
    extern __shared__ float shared_data[];

    // Move an element from the input array into shared data.
    // This thread block will perform the reduction over the shared data only, and then move that into the output.
    const uint t_id = threadIdx.x;
    const uint input_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Writing this into shared data is race-free, because each thread has a unique index.
    if (input_idx < input_length) {
        //if (blockIdx.x == 0)
        //    printf("%a\n", input[input_idx]);
        shared_data[t_id] = preprocess(input[input_idx]);
    } else {
        // Handle the case where the input data isn't pow2
        shared_data[t_id] = 0;
    }

    // __syncthreads is required before reading shared memory - while threads within a warp are all guaranteed to execute at the same time,
    // warps within a block are not. i.e. one warp could be started once another warp is waiting for a memory access.
    for (uint stride = 1; stride < blockDim.x; stride *= 2) {
        // Use a strided index - each thread reduces the indices (2*tid)*stride, (2*tid + 1)*stride
        const uint index = 2 * stride * t_id;

        // This is different to the PDF - I prefer to sync at the last possible second, to give as much leeway as possible to the scheduler beforehand,
        // and to make sure threads don't lose their sync inbetween the sync and the access.
        // This is done outside of the loop on purpose! Doing it inside a conditional is undefined behaviour.
        __syncthreads();
        if (index < blockDim.x) {
            shared_data[index] = f(shared_data[index], shared_data[index + stride]);
        }
    }

    if (t_id == 0)
        output[blockIdx.x] = shared_data[0];
}



// EXAMPLE DISPATCH

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
//template<size_t blockSize, typename T>
//T GPUReduction(T* dA, size_t N)
//{
//    T tot = 0.;
//    size_t n = N;
//    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);
//
//    T* tmp;
//    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");
//
//    T* from = dA;
//
//    do
//    {
//        blocksPerGrid   = std::ceil((1.*n) / blockSize);
//        reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
//        from = tmp;
//        n = blocksPerGrid;
//    } while (n > blockSize);
//
//    if (n > 1)
//        reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);
//
//    cudaDeviceSynchronize();
//    checkCUDAError("Error launching kernel [GPUReduction]");
//
//    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
//    cudaFree(tmp);
//    return tot;
//}

//template<size_t BlockSize, typename Preproc, typename Func>
//void dispatchReduction(in_matrix<float> input, size_t input_size, out_matrix<float> output,
//                       Preproc preproc, Func f, float null_data,
//
//                       cudaStream_t stream) {
//
//
//    size_t output_size = (input_size + BlockSize - 1) / BlockSize;
//    // output_size could be 3 i.e. if input_size = 192, BlockSize = 64
//    // round this up to a power of 2 so that the next iteration can use it https://jameshfisher.com/2018/03/30/round-up-power-2/
//    if (output_size != 1)
//        output_size = 1<<(64-__builtin_clzl(output_size-1));
//    while(true) {
//        dim3 blocksize = dim3(BlockSize);
//        // Each block produces exactly one output
//        dim3 gridsize = dim3(output_size);
//
//        reduce_simple<<<blocksize, gridsize, BlockSize*sizeof(float), stream>>>(
//                input, input_size,
//                output,
//
//                preproc, f, null_data
//        );
//
//        if (output_size == 1)
//            break; // We've just generated our output! hooray!
//
//        // If we need to keep going - the next iteration will produce exactly 1/2 the output
//    }
//}