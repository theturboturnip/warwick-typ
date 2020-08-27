//
// Created by samuel on 27/08/2020.
//

#pragma once

#if CUDA_ENABLED

#define CUDA_CHECK_IN_RELEASE 1
#if !NDEBUG
#define CUDA_CHECK_ENABLED 1
#elif CUDA_CHECK_IN_RELEASE
#define CUDA_CHECK_ENABLED 1
#else
#define CUDA_CHECK_ENABLED 0
#endif

#if CUDA_CHECK_ENABLED
#define CHECK_CUDA_ERROR(error) do { FATAL_ERROR_IF(error != cudaSuccess, "Cuda Error: %s\n", cudaGetErrorString(error)); } while(0);
#define CHECK_KERNEL_ERROR() CHECK_CUDA_ERROR(cudaPeekAtLastError())
#define CHECKED_CUDA(X) ([&]{ cudaError_t error = (X); CHECK_CUDA_ERROR(error); }())
#else
#define CHECK_CUDA_ERROR(error)
#define CHECK_KERNEL_ERROR()
#define CHECKED_CUDA(X) (X)
#endif

#endif