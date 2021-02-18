//
// Created by samuel on 18/02/2021.
//

#include "CudaGraphCapture.cuh"

CudaGraphCapture::~CudaGraphCapture() {
    if (graph.has_value() && graph.get()) {
        cudaGraphDestroy(graph.get());
        graph.set(nullptr);
    }
    if (instance.has_value() && instance.get()) {
        cudaGraphExecDestroy(instance.get());
        instance.set(nullptr);
    }
}

void CudaGraphCapture::recordOrExecute(std::function<void()> record) {
    if (!recorded) {
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
    cudaGraphLaunch(instance.get(), stream);
}
