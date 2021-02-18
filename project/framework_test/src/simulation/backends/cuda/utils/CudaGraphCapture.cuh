//
// Created by samuel on 18/02/2021.
//

#pragma once

#include <functional>

#include "util/ForgetOnMove.h"

class CudaGraphCapture {
    cudaStream_t stream;
    ForgetOnMove<cudaGraph_t> graph;
    ForgetOnMove<cudaGraphExec_t> instance;
    bool recorded = false;

public:
    explicit CudaGraphCapture(cudaStream_t stream) : stream(stream), graph(), instance(), recorded(false) {}
    CudaGraphCapture(CudaGraphCapture&&) noexcept = default;
    CudaGraphCapture(const CudaGraphCapture&) = delete;
    ~CudaGraphCapture();

    void recordOrExecute(std::function<void()> record);
};

