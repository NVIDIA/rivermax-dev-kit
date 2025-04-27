/*
 * Copyright © 2021-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its
 * affiliates (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */
#pragma once

// Concurrent FIFO.

//
// Copyright (c) 2013 Juan Palacios juan.palacios.puyana@gmail.com
// Subject to the BSD 2-Clause License
// - see < http://opensource.org/licenses/BSD-2-Clause>
//
// Adapted from: https://juanchopanzacpp.wordpress.com/2013/02/26/concurrent-queue-c11/

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template <typename T>
class Fifo
{
public:

    T pop()
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty())
        {
            if (interrupt_signal) {
                return nullptr;
            }
            cond_.wait(mlock);
        }
        auto item = queue_.front();
        queue_.pop();
        element_count--;
        return item;
    }

    void pop(T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty())
        {
            if (interrupt_signal) {
                return;
            }
            cond_.wait(mlock);
        }
        item = queue_.front();
        element_count--;
        queue_.pop();
    }

    void push(const T& item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push(item);
        element_count++;
        mlock.unlock();
        cond_.notify_one();
    }

    void push(T&& item)
    {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push(std::move(item));
        element_count++;
        mlock.unlock();
        cond_.notify_one();
    }

    void set_interrupt()
    {
        interrupt_signal = true;
        cond_.notify_all();
    }
 
    bool empty() const
    {
     //   std::unique_lock<std::mutex> mlock(mutex_);
        return queue_.empty();
    }

    unsigned int elementInQ() const
    {
     //   std::unique_lock<std::mutex> mlock(mutex_);
        return element_count;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    unsigned int element_count=0;
    bool interrupt_signal = false;
};
