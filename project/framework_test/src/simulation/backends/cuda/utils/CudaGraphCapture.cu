//
// Created by samuel on 18/02/2021.
//

#include "CudaGraphCapture.cuh"

CudaGraphCapture::~CudaGraphCapture() {
    if (graph.has_value() && graph.get()) {
        cudaGraphDestroy(graph.release());
    }
    if (instance.has_value() && instance.get()) {
        cudaGraphExecDestroy(instance.release());
    }
}

void CudaGraphCapture::record(std::function<void()> record) {
    DASSERT(!recorded);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    record();

    cudaGraph_t capturedGraph = nullptr;
    cudaStreamEndCapture(stream, &capturedGraph);
    DASSERT(capturedGraph != nullptr);

    graph.set(capturedGraph);

    cudaGraphExec_t toExec = nullptr;
    cudaGraphInstantiate(&toExec, graph.get(), nullptr, nullptr, 0);
    instance.set(toExec);

    recorded = true;
}

void CudaGraphCapture::execute() {
    cudaGraphLaunch(instance.get(), stream);
}
