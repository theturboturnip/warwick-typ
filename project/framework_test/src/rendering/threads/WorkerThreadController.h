//
// Created by samuel on 26/08/2020.
//

#pragma once

#include <thread>

#include "IWorkerThread.h"


/**
 * BaseThreads are the threading primitive used for work that's done per-frame.
 * Usage: call startWork with a single work item to dispatch asynchronously.
 * After you've done other things and expect the result, call getResult.
 * getResult will block if the thread isn't done yet.
 *
 * @tparam TFrameIn
 * @tparam TFrameOut
 */
template<class TFrameIn, class TFrameOut>
class WorkerThreadController {
    std::unique_ptr<IWorkerThread<TFrameIn, TFrameOut>> worker;
    std::thread workerThread;

    ThreadWorkData<TFrameIn>& inputDataRef;
    ThreadWorkData<TFrameOut>& outputDataRef;

    int32_t lastEnqueuedFrame = -1;
    bool waitingForWork = false;

    template<typename FuncType>
    void sendTransformedData(FuncType func) {
        lastEnqueuedFrame++;
        {
            std::lock_guard<std::mutex> lock(inputDataRef.sync.mutex);
            inputDataRef.index = lastEnqueuedFrame;
            func();
        }
        // In case someone is waiting on the condition variable, signal it
        inputDataRef.sync.readyForRead.notify_one();
        waitingForWork = true;
    }

public:
    explicit WorkerThreadController(std::unique_ptr<IWorkerThread<TFrameIn, TFrameOut>>&& worker)
        : worker(std::move(worker)),
          workerThread([this](){
              this->worker->threadLoop();
          }),
          inputDataRef(this->worker->inputData),
          outputDataRef(this->worker->outputData)
    {}
    ~WorkerThreadController() {
        if (waitingForWork)
            getOutput();
        // Send a message to the thread to tell it to join
        sendTransformedData([this]{
            inputDataRef.shouldJoin = true;
        });
        workerThread.join();
    }

    void giveNextWork(TFrameIn&& input) {
        sendTransformedData([this, input](){
            inputDataRef.data = input;
            inputDataRef.shouldJoin = false;
        });
    }

    TFrameOut&& getOutput() {
        std::unique_lock<std::mutex> lock(outputDataRef.sync.mutex);
        outputDataRef.sync.readyForRead.wait(lock, [&]{
          return outputDataRef.index == lastEnqueuedFrame;
        });
        // TODO - we're doing this inside the mutex?
        waitingForWork = false;
        return std::move(outputDataRef.data);
    }
};