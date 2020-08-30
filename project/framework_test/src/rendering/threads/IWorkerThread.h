//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "ThreadWorkData.h"

#include <optional>
#include <cstdint>

template<class TFrameIn, class TFrameOut>
class IWorkerThread {
public:
    ThreadWorkData<TFrameIn> inputData;
    ThreadWorkData<TFrameOut> outputData;

    virtual ~IWorkerThread() = default;
    virtual void threadLoop() = 0;
protected:
    int32_t currentWorkFrame = -1;

    std::optional<TFrameIn> waitForInput() {
        // unique_lock is used here because it can unlock/relock, which the condition variable needs.
        std::unique_lock<std::mutex> lock(inputData.sync.mutex);
        inputData.sync.readyForRead.wait(lock, [&]{
          return inputData.index > currentWorkFrame;
        });

        currentWorkFrame = inputData.index;
        if (inputData.shouldJoin)
            return std::nullopt;
        return std::move(inputData.data);
    }
    void pushOutput(TFrameOut&& output) {
        {
            std::lock_guard<std::mutex> lock(outputData.sync.mutex);
            outputData.index = currentWorkFrame;
            outputData.data = output;
        }
        // In case someone is waiting on the condition variable, signal it
        outputData.sync.readyForRead.notify_one();
    }
};