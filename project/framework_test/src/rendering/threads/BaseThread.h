//
// Created by samuel on 26/08/2020.
//

#pragma once

#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <thread>


template<class T>
struct ThreadWorkData {
    // Put the mutex and condition variables in their own cacheline
    struct alignas(64) {
        std::mutex mutex;
        std::condition_variable readyForRead;
    } sync;

    // Put the data in a separate cacheline, so that i.e. someone reading the condition variable doesn't try to use the same cacheline as someone writing to the index.
    struct alignas(64) {
        int32_t index = -1;
        bool shouldJoin = false;
        T data;
    };
};


template<class TFrameIn, class TFrameOut>
class IThreadWorker {
public:
    ThreadWorkData<TFrameIn> inputData;
    ThreadWorkData<TFrameOut> outputData;

    virtual ~IThreadWorker() = default;
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
class BaseThread {
    std::unique_ptr<IThreadWorker<TFrameIn, TFrameOut>> worker;
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
    explicit BaseThread(std::unique_ptr<IThreadWorker<TFrameIn, TFrameOut>>&& worker)
        : worker(std::move(worker)),
          workerThread([this](){
              this->worker->threadLoop();
          }),
          inputDataRef(this->worker->inputData),
          outputDataRef(this->worker->outputData)
    {}
    ~BaseThread() {
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