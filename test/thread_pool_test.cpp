/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../src/thread_pool.h"
#include "../src/imgproc/exception.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace nvimgcodec { namespace test {

namespace {

enum class CallerDeviceSwitchPoint
{
  kBeforeAddWork,
  kBeforeRun
};

void ExpectThreadPoolWorkersStayOnDevice0(CallerDeviceSwitchPoint switch_point) {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  int original_device = 0;
  CHECK_CUDA(cudaGetDevice(&original_device));

  const int pool_device = 0;
  const int caller_device = 1;
  const int num_threads = 4;

  std::atomic<int> ready{0};
  std::atomic<bool> release{false};
  std::mutex observed_mutex;
  std::vector<int> observed_devices;

  CHECK_CUDA(cudaSetDevice(pool_device));
  {
    ThreadPool tp(num_threads, pool_device, false, "ThreadPool device isolation test");

    if (switch_point == CallerDeviceSwitchPoint::kBeforeAddWork) {
      CHECK_CUDA(cudaSetDevice(caller_device));
    }

    for (int i = 0; i < num_threads; i++) {
      tp.addWork([&](int) {
        int current_device = -1;
        CHECK_CUDA(cudaGetDevice(&current_device));
        {
          std::lock_guard<std::mutex> lock(observed_mutex);
          observed_devices.push_back(current_device);
        }
        ready.fetch_add(1, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
      });
    }

    if (switch_point == CallerDeviceSwitchPoint::kBeforeRun) {
      CHECK_CUDA(cudaSetDevice(caller_device));
    }

    tp.run();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ready.load(std::memory_order_acquire) < num_threads &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    release.store(true, std::memory_order_release);
    tp.wait();
  }

  CHECK_CUDA(cudaSetDevice(original_device));

  ASSERT_EQ(ready.load(std::memory_order_acquire), num_threads);
  ASSERT_EQ(observed_devices.size(), static_cast<size_t>(num_threads));
  for (int device : observed_devices) {
    EXPECT_EQ(device, pool_device);
  }
}

}  // namespace

TEST(ThreadPool, AddWork) {
  ThreadPool tp(16, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.addWork(increase);
  }
  ASSERT_EQ(count, 0);
  tp.run();
  tp.wait();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, AddWorkImmediateStart) {
  ThreadPool tp(16, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.addWork(increase);
    tp.run();
  }
  tp.wait();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, WorkerThreadsKeepRequestedDeviceCurrent_MultiGPU) {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  int original_device = 0;
  CHECK_CUDA(cudaGetDevice(&original_device));

  const int target_device = 1;
  const int num_threads = 4;
  ThreadPool tp(num_threads, target_device, false, "ThreadPool device test");

  std::atomic<int> ready{0};
  std::atomic<bool> release{false};
  std::mutex observed_mutex;
  std::vector<int> observed_devices;

  for (int i = 0; i < num_threads; i++) {
    tp.addWork([&](int) {
      int current_device = -1;
      CHECK_CUDA(cudaGetDevice(&current_device));
      {
        std::lock_guard<std::mutex> lock(observed_mutex);
        observed_devices.push_back(current_device);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    });
  }

  tp.run();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (ready.load(std::memory_order_acquire) < num_threads &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  release.store(true, std::memory_order_release);
  tp.wait();

  CHECK_CUDA(cudaSetDevice(original_device));

  ASSERT_EQ(ready.load(std::memory_order_acquire), num_threads);
  ASSERT_EQ(observed_devices.size(), static_cast<size_t>(num_threads));
  for (int device : observed_devices) {
    EXPECT_EQ(device, target_device);
  }
}

TEST(ThreadPool, WorkerThreadsIgnoreCallerDeviceSwitchBeforeAddWork_MultiGPU) {
  ExpectThreadPoolWorkersStayOnDevice0(CallerDeviceSwitchPoint::kBeforeAddWork);
}

TEST(ThreadPool, WorkerThreadsIgnoreCallerDeviceSwitchBeforeRun_MultiGPU) {
  ExpectThreadPoolWorkersStayOnDevice0(CallerDeviceSwitchPoint::kBeforeRun);
}

}  // namespace test

} // namespace nvimgcodec
