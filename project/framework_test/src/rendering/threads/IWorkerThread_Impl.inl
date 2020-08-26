//
// Created by samuel on 26/08/2020.
//

#pragma once

#include "IWorkerThread.h"

template<class Worker, class TFrameIn, class TFrameOut>
class IWorkerThread_Impl : public IWorkerThread<TFrameIn, TFrameOut> {
    Worker worker;

public:
    template<typename... Args>
    IWorkerThread_Impl(Args&&... args) : worker(std::forward<Args&&...>(args...)) {}
    ~IWorkerThread_Impl() override = default;

    void threadLoop() override {
        while (true) {
            const auto input = this->waitForInput();
            if (!input.has_value())
                break;

            this->pushOutput(worker.work(input.value()));
        }
    }
};