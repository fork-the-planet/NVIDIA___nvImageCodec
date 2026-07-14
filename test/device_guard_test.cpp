/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>
#include "../src/imgproc/device_guard.h"
#include "../src/imgproc/device_buffer.h"
#include "../src/imgproc/exception.h"
#include "../src/imgproc/pinned_buffer.h"
#include "../src/imgproc/stream_device.h"
#include "parsers/parser_test_utils.h"

namespace nvimgcodec {
namespace test {

namespace {

unsigned char kSmallBmp[] = {0x42, 0x4D, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00};

constexpr unsigned char kDecodedSentinel = 0x5a;

bool HasMultiGpu()
{
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  return count >= 2;
}

int CurrentCudaDevice()
{
  int device = -1;
  CHECK_CUDA(cudaGetDevice(&device));
  return device;
}

int PointerDevice(const void* ptr)
{
  cudaPointerAttributes attr{};
  CHECK_CUDA(cudaPointerGetAttributes(&attr, ptr));
  return attr.device;
}

struct ImageCudaPlacement
{
  bool image_info_valid = false;
  nvimgcodecImageBufferKind_t buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN;
  cudaStream_t stream = nullptr;
  void* buffer = nullptr;
  int stream_device = -1;
  int buffer_device = -1;
};

ImageCudaPlacement InspectImageCudaPlacement(const nvimgcodecImageDesc_t* image)
{
  ImageCudaPlacement placement;
  nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
  if (image->getImageInfo(image->instance, &image_info) != NVIMGCODEC_STATUS_SUCCESS)
    return placement;

  placement.image_info_valid = true;
  placement.buffer_kind = image_info.buffer_kind;
  placement.stream = image_info.cuda_stream;
  placement.buffer = image_info.buffer;
  try {
    placement.stream_device = get_stream_device_id(image_info.cuda_stream);
  } catch (...) {
    placement.stream_device = -1;
  }

  if (image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE && image_info.buffer) {
    cudaPointerAttributes attr{};
    auto err = cudaPointerGetAttributes(&attr, image_info.buffer);
    if (err == cudaSuccess) {
      placement.buffer_device = attr.device;
    } else {
      (void)cudaGetLastError();
    }
  }
  return placement;
}

bool PlacementMatchesDevice(const ImageCudaPlacement& placement, int expected_device)
{
  return placement.image_info_valid && placement.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE &&
         placement.stream_device == expected_device && placement.buffer_device == expected_device;
}

nvimgcodecImageInfo_t MakeHostRgbImageInfo(void* buffer)
{
  nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
  std::strcpy(image_info.codec_name, "bmp");
  image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
  image_info.num_planes = 1;
  image_info.plane_info[0] = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO, sizeof(nvimgcodecImagePlaneInfo_t), nullptr};
  image_info.plane_info[0].width = 1;
  image_info.plane_info[0].height = 1;
  image_info.plane_info[0].num_channels = 3;
  image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  image_info.plane_info[0].precision = 8;
  image_info.plane_info[0].row_stride = 3;
  image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
  image_info.buffer = buffer;
  return image_info;
}

class DeviceCheckingDecoderPlugin
{
 public:
  explicit DeviceCheckingDecoderPlugin(int expected_device)
      : expected_device_(expected_device)
      , decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), nullptr, this, "device_guard_decoder", "bmp",
            NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, static_destroy, static_get_metadata, static_can_decode, static_decode_sample,
            nullptr, nullptr}
  {}

  nvimgcodecDecoderDesc_t* getDecoderDesc() { return &decoder_desc_; }

  bool AllObservedDevicesMatch() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return create_device_ == expected_device_ && destroy_device_ == expected_device_ &&
           std::all_of(can_decode_devices_.begin(), can_decode_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(decode_devices_.begin(), decode_devices_.end(), [&](int dev) { return dev == expected_device_; });
  }

  bool LazyObservedDevicesMatch() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return create_device_ == expected_device_ && metadata_device_ == expected_device_ &&
           !can_decode_devices_.empty() &&
           std::all_of(can_decode_devices_.begin(), can_decode_devices_.end(), [&](int dev) { return dev == expected_device_; });
  }

  bool DecodeObservedDevicesMatch() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return !decode_devices_.empty() && !can_decode_devices_.empty() && decode_stream_devices_.size() == decode_devices_.size() &&
           decode_buffer_devices_.size() == decode_devices_.size() &&
           std::all_of(can_decode_devices_.begin(), can_decode_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(decode_devices_.begin(), decode_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(decode_stream_devices_.begin(), decode_stream_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(decode_buffer_devices_.begin(), decode_buffer_devices_.end(), [&](int dev) { return dev == expected_device_; });
  }

 private:
  static nvimgcodecStatus_t static_create(
      void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
  {
    auto* handle = reinterpret_cast<DeviceCheckingDecoderPlugin*>(instance);
    std::lock_guard<std::mutex> lock(handle->mutex_);
    handle->create_device_ = CurrentCudaDevice();
    *decoder = reinterpret_cast<nvimgcodecDecoder_t>(handle);
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder)
  {
    auto* handle = reinterpret_cast<DeviceCheckingDecoderPlugin*>(decoder);
    std::lock_guard<std::mutex> lock(handle->mutex_);
    handle->destroy_device_ = CurrentCudaDevice();
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecStatus_t static_get_metadata(
      nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count)
  {
    auto* handle = reinterpret_cast<DeviceCheckingDecoderPlugin*>(decoder);
    std::lock_guard<std::mutex> lock(handle->mutex_);
    handle->metadata_device_ = CurrentCudaDevice();
    if (metadata_count)
      *metadata_count = 0;
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
      const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
  {
    auto* handle = reinterpret_cast<DeviceCheckingDecoderPlugin*>(decoder);
    int device = CurrentCudaDevice();
    {
      std::lock_guard<std::mutex> lock(handle->mutex_);
      handle->can_decode_devices_.push_back(device);
    }
    return device == handle->expected_device_ ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS : NVIMGCODEC_PROCESSING_STATUS_FAIL;
  }

  static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
      const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
  {
    auto* handle = reinterpret_cast<DeviceCheckingDecoderPlugin*>(decoder);
    int device = CurrentCudaDevice();
    auto placement = InspectImageCudaPlacement(image);
    auto status = device == handle->expected_device_ && PlacementMatchesDevice(placement, handle->expected_device_)
                      ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS
                      : NVIMGCODEC_PROCESSING_STATUS_FAIL;
    if (status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
      auto err = cudaMemsetAsync(placement.buffer, kDecodedSentinel, 1, placement.stream);
      if (err != cudaSuccess) {
        (void)cudaGetLastError();
        status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
      }
    }
    {
      std::lock_guard<std::mutex> lock(handle->mutex_);
      handle->decode_devices_.push_back(device);
      handle->decode_stream_devices_.push_back(placement.stream_device);
      handle->decode_buffer_devices_.push_back(placement.buffer_device);
    }
    image->imageReady(image->instance, status);
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  int expected_device_ = 0;
  nvimgcodecDecoderDesc_t decoder_desc_{};
  mutable std::mutex mutex_;
  int create_device_ = -1;
  int destroy_device_ = -1;
  int metadata_device_ = -1;
  std::vector<int> can_decode_devices_;
  std::vector<int> decode_devices_;
  std::vector<int> decode_stream_devices_;
  std::vector<int> decode_buffer_devices_;
};

class DeviceCheckingEncoderPlugin
{
 public:
  explicit DeviceCheckingEncoderPlugin(int expected_device)
      : expected_device_(expected_device)
      , encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), nullptr, this, "device_guard_encoder", "bmp",
            NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, static_destroy, static_can_encode, static_encode_sample}
  {}

  nvimgcodecEncoderDesc_t* getEncoderDesc() { return &encoder_desc_; }

  bool LazyObservedDevicesMatch() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return create_device_ == expected_device_ && !can_encode_devices_.empty() &&
           std::all_of(can_encode_devices_.begin(), can_encode_devices_.end(), [&](int dev) { return dev == expected_device_; });
  }

  bool DestroyObservedDeviceMatches() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return create_device_ == expected_device_ && destroy_device_ == expected_device_;
  }

  bool EncodeObservedDevicesMatch() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return !can_encode_devices_.empty() && !encode_devices_.empty() && encode_stream_devices_.size() == encode_devices_.size() &&
           encode_buffer_devices_.size() == encode_devices_.size() &&
           std::all_of(can_encode_devices_.begin(), can_encode_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(encode_devices_.begin(), encode_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(encode_stream_devices_.begin(), encode_stream_devices_.end(), [&](int dev) { return dev == expected_device_; }) &&
           std::all_of(encode_buffer_devices_.begin(), encode_buffer_devices_.end(), [&](int dev) { return dev == expected_device_; });
  }

 private:
  static nvimgcodecStatus_t static_create(
      void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
  {
    auto* handle = reinterpret_cast<DeviceCheckingEncoderPlugin*>(instance);
    std::lock_guard<std::mutex> lock(handle->mutex_);
    handle->create_device_ = CurrentCudaDevice();
    *encoder = reinterpret_cast<nvimgcodecEncoder_t>(handle);
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder)
  {
    auto* handle = reinterpret_cast<DeviceCheckingEncoderPlugin*>(encoder);
    std::lock_guard<std::mutex> lock(handle->mutex_);
    handle->destroy_device_ = CurrentCudaDevice();
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
      const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
  {
    auto* handle = reinterpret_cast<DeviceCheckingEncoderPlugin*>(encoder);
    int device = CurrentCudaDevice();
    {
      std::lock_guard<std::mutex> lock(handle->mutex_);
      handle->can_encode_devices_.push_back(device);
    }
    return device == handle->expected_device_ ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS : NVIMGCODEC_PROCESSING_STATUS_FAIL;
  }

  static nvimgcodecStatus_t static_encode_sample(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
      const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
  {
    auto* handle = reinterpret_cast<DeviceCheckingEncoderPlugin*>(encoder);
    int device = CurrentCudaDevice();
    auto placement = InspectImageCudaPlacement(image);
    auto status = device == handle->expected_device_ && PlacementMatchesDevice(placement, handle->expected_device_)
                      ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS
                      : NVIMGCODEC_PROCESSING_STATUS_FAIL;
    if (status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
      auto err = cudaMemsetAsync(placement.buffer, 0, 1, placement.stream);
      if (err != cudaSuccess) {
        (void)cudaGetLastError();
        status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
      }
    }
    {
      std::lock_guard<std::mutex> lock(handle->mutex_);
      handle->encode_devices_.push_back(device);
      handle->encode_stream_devices_.push_back(placement.stream_device);
      handle->encode_buffer_devices_.push_back(placement.buffer_device);
    }
    image->imageReady(image->instance, status);
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  int expected_device_ = 0;
  nvimgcodecEncoderDesc_t encoder_desc_{};
  mutable std::mutex mutex_;
  int create_device_ = -1;
  int destroy_device_ = -1;
  std::vector<int> can_encode_devices_;
  std::vector<int> encode_devices_;
  std::vector<int> encode_stream_devices_;
  std::vector<int> encode_buffer_devices_;
};

class DeviceCheckingExtensionFactory
{
 public:
  explicit DeviceCheckingExtensionFactory(int expected_device)
      : decoder_(expected_device)
      , encoder_(expected_device)
      , desc_{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, this, "device_guard_extension",
            NVIMGCODEC_VER, NVIMGCODEC_VER, static_extension_create, static_extension_destroy}
  {}

  nvimgcodecExtensionDesc_t* getExtensionDesc() { return &desc_; }
  DeviceCheckingDecoderPlugin& decoder() { return decoder_; }
  DeviceCheckingEncoderPlugin& encoder() { return encoder_; }

 private:
  struct Extension
  {
    Extension(DeviceCheckingExtensionFactory* owner, const nvimgcodecFrameworkDesc_t* framework)
        : owner_(owner)
        , framework_(framework)
    {
      framework_->registerDecoder(framework_->instance, owner_->decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGHEST);
      framework_->registerEncoder(framework_->instance, owner_->encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_HIGHEST);
    }

    ~Extension()
    {
      framework_->unregisterEncoder(framework_->instance, owner_->encoder_.getEncoderDesc());
      framework_->unregisterDecoder(framework_->instance, owner_->decoder_.getDecoderDesc());
    }

    DeviceCheckingExtensionFactory* owner_;
    const nvimgcodecFrameworkDesc_t* framework_;
  };

  static nvimgcodecStatus_t static_extension_create(
      void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
  {
    auto* handle = reinterpret_cast<DeviceCheckingExtensionFactory*>(instance);
    *extension = reinterpret_cast<nvimgcodecExtension_t>(new Extension(handle, framework));
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecStatus_t static_extension_destroy(nvimgcodecExtension_t extension)
  {
    delete reinterpret_cast<Extension*>(extension);
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  DeviceCheckingDecoderPlugin decoder_;
  DeviceCheckingEncoderPlugin encoder_;
  nvimgcodecExtensionDesc_t desc_{};
};

struct WrongDeviceExecutor
{
  struct Task
  {
    int sample_idx = -1;
    void* context = nullptr;
    void (*task)(int thread_id, int sample_idx, void* task_context) = nullptr;
  };

  WrongDeviceExecutor(int num_threads, int wrong_device)
      : num_threads_(num_threads)
      , wrong_device_(wrong_device)
      , desc_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, sizeof(nvimgcodecExecutorDesc_t), nullptr, this, static_schedule, static_run,
            static_wait, static_get_num_threads}
  {}

  ~WrongDeviceExecutor() { wait(); }

  nvimgcodecExecutorDesc_t* getDesc() { return &desc_; }

  void wait()
  {
    for (auto& thread : running_) {
      if (thread.joinable())
        thread.join();
    }
    running_.clear();
  }

  static nvimgcodecStatus_t static_schedule(
      void* instance, int device_id, int sample_idx, void* task_context, void (*task)(int thread_id, int sample_idx, void* task_context))
  {
    auto* handle = reinterpret_cast<WrongDeviceExecutor*>(instance);
    handle->pending_.push_back(Task{sample_idx, task_context, task});
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecStatus_t static_run(void* instance, int device_id)
  {
    auto* handle = reinterpret_cast<WrongDeviceExecutor*>(instance);
    handle->wait();
    std::vector<Task> tasks;
    tasks.swap(handle->pending_);
    for (size_t i = 0; i < tasks.size(); i++) {
      handle->running_.emplace_back([handle, task = tasks[i], tid = static_cast<int>(i)] {
        CHECK_CUDA(cudaSetDevice(handle->wrong_device_));
        task.task(tid, task.sample_idx, task.context);
      });
    }
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static nvimgcodecStatus_t static_wait(void* instance, int device_id)
  {
    auto* handle = reinterpret_cast<WrongDeviceExecutor*>(instance);
    handle->wait();
    return NVIMGCODEC_STATUS_SUCCESS;
  }

  static int static_get_num_threads(void* instance)
  {
    auto* handle = reinterpret_cast<WrongDeviceExecutor*>(instance);
    return handle->num_threads_;
  }

  int num_threads_;
  int wrong_device_;
  nvimgcodecExecutorDesc_t desc_{};
  std::vector<Task> pending_;
  std::vector<std::thread> running_;
};

struct AllocatorDeviceRecorder
{
  int device_malloc_device = -1;
  int device_malloc_pointer_device = -1;
  int device_free_device = -1;
  int device_free_pointer_device = -1;
  int pinned_malloc_device = -1;
  int pinned_free_device = -1;

  static int device_malloc(void* ctx, void** ptr, size_t size, cudaStream_t stream)
  {
    auto* recorder = reinterpret_cast<AllocatorDeviceRecorder*>(ctx);
    recorder->device_malloc_device = CurrentCudaDevice();
    auto err = cudaMalloc(ptr, size);
    if (err != cudaSuccess)
      return 1;
    recorder->device_malloc_pointer_device = PointerDevice(*ptr);
    return 0;
  }

  static int device_free(void* ctx, void* ptr, size_t size, cudaStream_t stream)
  {
    auto* recorder = reinterpret_cast<AllocatorDeviceRecorder*>(ctx);
    recorder->device_free_device = CurrentCudaDevice();
    recorder->device_free_pointer_device = PointerDevice(ptr);
    return cudaFree(ptr) == cudaSuccess ? 0 : 1;
  }

  static int pinned_malloc(void* ctx, void** ptr, size_t size, cudaStream_t stream)
  {
    auto* recorder = reinterpret_cast<AllocatorDeviceRecorder*>(ctx);
    recorder->pinned_malloc_device = CurrentCudaDevice();
    return cudaMallocHost(ptr, size) == cudaSuccess ? 0 : 1;
  }

  static int pinned_free(void* ctx, void* ptr, size_t size, cudaStream_t stream)
  {
    auto* recorder = reinterpret_cast<AllocatorDeviceRecorder*>(ctx);
    recorder->pinned_free_device = CurrentCudaDevice();
    return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
  }
};

struct FailingPinnedAllocator
{
  bool fail_next_alloc = false;

  static int pinned_malloc(void* ctx, void** ptr, size_t size, cudaStream_t stream)
  {
    auto* allocator = reinterpret_cast<FailingPinnedAllocator*>(ctx);
    if (allocator->fail_next_alloc)
      return 1;
    return cudaMallocHost(ptr, size) == cudaSuccess ? 0 : 1;
  }

  static int pinned_free(void* ctx, void* ptr, size_t size, cudaStream_t stream)
  {
    return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
  }
};

struct FailingDeviceAllocator
{
  bool fail_next_alloc = false;

  static int device_malloc(void* ctx, void** ptr, size_t size, cudaStream_t stream)
  {
    auto* allocator = reinterpret_cast<FailingDeviceAllocator*>(ctx);
    if (allocator->fail_next_alloc)
      return 1;
    return cudaMalloc(ptr, size) == cudaSuccess ? 0 : 1;
  }

  static int device_free(void* ctx, void* ptr, size_t size, cudaStream_t stream)
  {
    return cudaFree(ptr) == cudaSuccess ? 0 : 1;
  }
};

struct FreeOnlyDeviceAllocatorRecorder
{
  bool device_free_called = false;

  static int device_free(void* ctx, void* ptr, size_t size, cudaStream_t stream)
  {
    auto* recorder = reinterpret_cast<FreeOnlyDeviceAllocatorRecorder*>(ctx);
    recorder->device_free_called = true;
    return cudaFree(ptr) == cudaSuccess ? 0 : 1;
  }
};

}  // namespace

TEST(DeviceGuard, ConstructorWithDevice) {
  int test_device = 0;
  int guard_device = 0;
  int current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CUDA(cudaSetDevice(test_device));
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g(guard_device);
    CHECK_CUDA(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
  }
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
}

TEST(DeviceGuard, ConstructorNoArgs) {
  int test_device = 0;
  int guard_device = 0;
  int current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CUDA(cudaSetDevice(test_device));
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g;
    CHECK_CUDA(cudaSetDevice(guard_device));
    CHECK_CUDA(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
  }
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
}

namespace {

struct CUDAContext {
  CUDAContext(CUcontext handle):handle_(handle){}
  inline ~CUDAContext() { DestroyHandle(handle_); }

  static CUDAContext Create(int flags, CUdevice dev) {
    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CUctxCreateParams params = {};
    CHECK_CU(cuCtxCreate(&ctx, &params, 0, dev));
#else
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
#endif
    return CUDAContext(ctx);
  }

  static void DestroyHandle(CUcontext ctx) {
    CHECK_CU(cuCtxDestroy(ctx));
  }
  CUcontext handle_;
};

// Subprocess body: with no primary context active for device 0, a DeviceGuard
// constructed for device 1 must capture a NULL "uninitialized" context and
// restore it faithfully on destruction (regression for the prior fix where
// NULL was used as both sentinel and value).
int DoTestUninit0() {
  if (!cuInitChecked()) {
    std::cout << "Cannot initialize CUDA." << std::endl;
    return 0xdead;
  }

  unsigned flags = 0;
  int active = 0;
  CHECK_CU(cuDevicePrimaryCtxGetState(0, &flags, &active));
  if (active) {
    std::cout << "This test cannot be run with a primary context already active for device 0"
              << std::endl;
    return 0xbad;
  }

  int curr = -1;
  {
    DeviceGuard dg(1);
    CHECK_CUDA(cudaGetDevice(&curr));
    if (curr != 1) {
      std::cout << "Device not switched properly." << std::endl;
      return 1;
    }
  }
  CHECK_CUDA(cudaGetDevice(&curr));
  if (curr != 0) {
    std::cout << "Device not restored properly." << std::endl;
    return 2;
  }
  return 0;
}

void TestUninit0() {
  std::_Exit(DoTestUninit0());
}

// Subprocess body: many threads racing on DeviceGuard(0) must end with the
// device's primary context retained exactly once (lazy init is thread-safe).
int DoTestPrimaryContextInitMultithreaded() {
  if (!cuInitChecked()) {
    std::cout << "Cannot initialize CUDA." << std::endl;
    return 0xdead;
  }

  unsigned flags = 0;
  int active = 0;
  CHECK_CU(cuDevicePrimaryCtxGetState(0, &flags, &active));
  if (active) {
    std::cout << "This test cannot be run with a primary context already active for device 0"
              << std::endl;
    return 0xbad;
  }

  std::atomic<bool> start{false};
  std::atomic<bool> failure{false};

  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&]() {
      while (!start.load()) {}
      DeviceGuard dg(0);
      unsigned f = 0;
      int a = 0;
      if (cuDevicePrimaryCtxGetState(0, &f, &a) != CUDA_SUCCESS) {
        std::cout << "Cannot get primary context for device 0." << std::endl;
        failure = true;
      }
      if (!a) {
        std::cout << "Primary context not active" << std::endl;
        failure = true;
      }
    });
  }

  start.store(true);
  for (auto &t : threads)
    t.join();

  if (failure) {
    std::cout << "A failure was detected in one of the worker threads." << std::endl;
    return 1;
  }

  unsigned final_flags = 0;
  int final_active = 0;
  if (cuDevicePrimaryCtxGetState(0, &final_flags, &final_active) != CUDA_SUCCESS) {
    std::cout << "Cannot get primary context for device 0." << std::endl;
    return 2;
  }
  if (!final_active) {
    std::cout << "Primary context should be active as a side-effect" << std::endl;
    return 3;
  }
  return 0;
}

void TestPrimaryContextInitMultithreaded() {
  std::_Exit(DoTestPrimaryContextInitMultithreaded());
}

}  // namespace

TEST(DeviceGuard, RestoreUninit0_MultiGPU) {
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(TestUninit0(), testing::ExitedWithCode(0), "");
}

TEST(DeviceGuard, Multithreaded) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(TestPrimaryContextInitMultithreaded(), testing::ExitedWithCode(0), "");
}

TEST(DeviceGuard, CheckContext_MultiGPU) {
  ASSERT_TRUE(cuInitChecked());

  int ndevs = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndevs));
  if (ndevs < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  int test_dev = ndevs - 1;
  CUcontext old = nullptr;
  CHECK_CU(cuCtxGetCurrent(&old));
  // Restore whatever was current when the test entered.
  struct Restore {
    CUcontext ctx;
    ~Restore() {
      CUresult err = cuCtxSetCurrent(ctx);
      if (err != CUDA_SUCCESS) {
        std::cerr << "cuCtxSetCurrent failed: " << err << std::endl;
      }
    }
  } restore{old};

  auto cu_test_ctx0 = CUDAContext::Create(0, 0);
  auto cu_test_ctx1 = CUDAContext::Create(0, 0);
  CHECK_CU(cuCtxSetCurrent(cu_test_ctx0.handle_));
  CUcontext ctx = nullptr;
  {
    DeviceGuard g;
    CHECK_CU(cuCtxGetCurrent(&ctx));
    EXPECT_EQ(ctx, cu_test_ctx0.handle_);
    CHECK_CU(cuCtxSetCurrent(cu_test_ctx1.handle_));
  }
  CHECK_CU(cuCtxGetCurrent(&ctx));
  EXPECT_EQ(ctx, cu_test_ctx0.handle_) << "Context not restored upon destruction";
  {
    DeviceGuard g(test_dev);
    CHECK_CU(cuCtxGetCurrent(&ctx));
    EXPECT_NE(ctx, cu_test_ctx0.handle_);
    CUdevice dev = -1;
    CHECK_CU(cuCtxGetDevice(&dev));
    EXPECT_EQ(dev, test_dev);
    CHECK_CU(cuCtxSetCurrent(cu_test_ctx1.handle_));
  }
  CHECK_CU(cuCtxGetCurrent(&ctx));
  EXPECT_EQ(ctx, cu_test_ctx0.handle_) << "Context not restored upon destruction";
}

TEST(DeviceGuard, CheckContextNoArgs) {
  int test_device = 0;
  CUdevice cu_test_device = 0;
  CUcontext cu_current_ctx = nullptr;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CU(cuDeviceGet(&cu_test_device, test_device));
  auto cu_test_ctx = CUDAContext::Create(0, cu_test_device);

  CHECK_CU(cuCtxSetCurrent(cu_test_ctx.handle_));
  CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx.handle_);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g;
    CHECK_CUDA(cudaSetDevice(guard_device));
    CHECK_CUDA(cudaGetDevice(&current_device));
    CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
    EXPECT_NE(cu_current_ctx, cu_test_ctx.handle_);
  }
  CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx.handle_);
  EXPECT_EQ(cu_current_device, cu_test_device);
}

TEST(DeviceGuard, NegativeDeviceIsNoOp) {
  CUdevice cu_test_device = 0;
  CUdevice cu_current_device = 0;
  CUcontext current_context = nullptr;

  ASSERT_TRUE(cuInitChecked());

  // Outer guard restores whatever device/context were current when this test
  // entered, so we can freely clobber them below.
  DeviceGuard restore_guard;
  CHECK_CU(cuDeviceGet(&cu_test_device, 0));
  auto test_context = CUDAContext::Create(0, cu_test_device);
  CHECK_CU(cuCtxSetCurrent(test_context.handle_));
  CHECK_CU(cuCtxGetCurrent(&current_context));
  ASSERT_EQ(current_context, test_context.handle_);

  {
    DeviceGuard g(-1);
  }

  CHECK_CU(cuCtxGetCurrent(&current_context));
  EXPECT_EQ(current_context, test_context.handle_);
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_device, cu_test_device);
}

TEST(DeviceGuard, NegativeDeviceNoOp_MultiGPU) {
  // Create a DeviceGuard with a negative device ID and destroy it out-of-order
  // to verify it is really a no-op even when a valid context is current.
  ASSERT_TRUE(cuInitChecked());
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  int dev = -1;
  {
    DeviceGuard dg0(0);
    std::optional<DeviceGuard> dgneg;
    {
      DeviceGuard dg1(1);
      dgneg.emplace(-1);  // outlives this scope
      CHECK_CUDA(cudaGetDevice(&dev));
      EXPECT_EQ(dev, 1);
    }
    CHECK_CUDA(cudaGetDevice(&dev));
    EXPECT_EQ(dev, 0);
    dgneg.reset();  // shouldn't change anything
    CHECK_CUDA(cudaGetDevice(&dev));
    EXPECT_EQ(dev, 0);
  }
  CHECK_CUDA(cudaGetDevice(&dev));
  EXPECT_EQ(dev, 0);
}

TEST(DeviceGuard, GenericCanDecodeAndMetadataUseRequestedDevice_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  constexpr int wrong_device = 0;
  constexpr int target_device = 1;
  CHECK_CUDA(cudaSetDevice(wrong_device));

  DeviceCheckingExtensionFactory extension_factory(target_device);
  nvimgcodecInstance_t instance = nullptr;
  nvimgcodecExtension_t extension = nullptr;
  nvimgcodecDecoder_t decoder = nullptr;
  nvimgcodecCodeStream_t code_stream = nullptr;
  nvimgcodecImage_t image = nullptr;
  std::vector<unsigned char> out_buffer(3);

  nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
  create_info.load_builtin_modules = 1;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, extension_factory.getExtensionDesc()));

  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = target_device;
  exec_params.max_num_cpu_threads = 1;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

  LoadImageFromHostMemory(instance, code_stream, kSmallBmp, sizeof(kSmallBmp));
  auto image_info = MakeHostRgbImageInfo(out_buffer.data());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &image, &image_info));

  nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
  nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(decoder, &code_stream, &image, 1, &params, &status, 1));
  EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

  int metadata_count = -1;
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderGetMetadata(decoder, code_stream, nullptr, &metadata_count));
  EXPECT_EQ(0, metadata_count);
  EXPECT_TRUE(extension_factory.decoder().LazyObservedDevicesMatch());

  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(DeviceGuard, GenericCanEncodeUsesRequestedDevice_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  constexpr int wrong_device = 0;
  constexpr int target_device = 1;
  CHECK_CUDA(cudaSetDevice(wrong_device));

  DeviceCheckingExtensionFactory extension_factory(target_device);
  nvimgcodecInstance_t instance = nullptr;
  nvimgcodecExtension_t extension = nullptr;
  nvimgcodecEncoder_t encoder = nullptr;
  nvimgcodecCodeStream_t code_stream = nullptr;
  nvimgcodecImage_t image = nullptr;
  std::vector<unsigned char> in_buffer(3);
  std::vector<unsigned char> out_buffer;

  nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, extension_factory.getExtensionDesc()));

  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = target_device;
  exec_params.max_num_cpu_threads = 1;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr));

  auto image_info = MakeHostRgbImageInfo(in_buffer.data());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &image, &image_info));
  nvimgcodecImageInfo_t out_image_info = image_info;
  out_image_info.buffer = nullptr;
  auto resize_output = [](void* ctx, size_t bytes) -> unsigned char* {
    auto* buffer = reinterpret_cast<std::vector<unsigned char>*>(ctx);
    buffer->resize(bytes);
    return buffer->data();
  };
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance, &code_stream, &out_buffer, resize_output, &out_image_info));

  nvimgcodecEncodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr};
  nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCanEncode(encoder, &image, &code_stream, 1, &params, &status, 1));
  EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
  EXPECT_TRUE(extension_factory.encoder().LazyObservedDevicesMatch());

  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(DeviceGuard, GenericDestroyUsesRequestedDevice_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  constexpr int wrong_device = 0;
  constexpr int target_device = 1;
  CHECK_CUDA(cudaSetDevice(wrong_device));

  DeviceCheckingExtensionFactory extension_factory(target_device);
  nvimgcodecInstance_t instance = nullptr;
  nvimgcodecExtension_t extension = nullptr;
  nvimgcodecDecoder_t decoder = nullptr;
  nvimgcodecEncoder_t encoder = nullptr;

  nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, extension_factory.getExtensionDesc()));

  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = target_device;
  exec_params.max_num_cpu_threads = 1;
  exec_params.pre_init = 1;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr));

  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder));
  EXPECT_TRUE(extension_factory.decoder().AllObservedDevicesMatch());
  EXPECT_TRUE(extension_factory.encoder().DestroyObservedDeviceMatches());

  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(DeviceGuard, CustomExecutorCallbacksUseRequestedDevice_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  constexpr int wrong_device = 0;
  constexpr int target_device = 1;
  CHECK_CUDA(cudaSetDevice(wrong_device));

  DeviceCheckingExtensionFactory extension_factory(target_device);
  WrongDeviceExecutor executor(2, wrong_device);
  nvimgcodecInstance_t instance = nullptr;
  nvimgcodecExtension_t extension = nullptr;
  nvimgcodecDecoder_t decoder = nullptr;
  std::vector<nvimgcodecCodeStream_t> code_streams(3, nullptr);
  std::vector<nvimgcodecImage_t> images(3, nullptr);
  std::vector<std::vector<unsigned char>> out_buffers(3, std::vector<unsigned char>(3));

  nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
  create_info.load_builtin_modules = 1;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, extension_factory.getExtensionDesc()));

  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = target_device;
  exec_params.max_num_cpu_threads = 2;
  exec_params.executor = executor.getDesc();
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

  for (size_t i = 0; i < code_streams.size(); i++) {
    LoadImageFromHostMemory(instance, code_streams[i], kSmallBmp, sizeof(kSmallBmp));
    auto image_info = MakeHostRgbImageInfo(out_buffers[i].data());
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &images[i], &image_info));
  }

  nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
  nvimgcodecFuture_t future = nullptr;
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder, code_streams.data(), images.data(), code_streams.size(), &params, &future));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future));
  executor.wait();
  std::vector<nvimgcodecProcessingStatus_t> statuses(code_streams.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
  size_t statuses_size = 0;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &statuses_size));
  ASSERT_EQ(code_streams.size(), statuses_size);
  for (auto status : statuses)
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
  EXPECT_TRUE(extension_factory.decoder().DecodeObservedDevicesMatch());
  for (const auto& out_buffer : out_buffers)
    EXPECT_EQ(kDecodedSentinel, out_buffer[0]);
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future));

  for (auto image : images)
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
  for (auto code_stream : code_streams)
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(DeviceGuard, CustomExecutorEncodeUsesRequestedDeviceAndDeviceBuffer_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  constexpr int wrong_device = 0;
  constexpr int target_device = 1;
  CHECK_CUDA(cudaSetDevice(wrong_device));

  DeviceCheckingExtensionFactory extension_factory(target_device);
  WrongDeviceExecutor executor(2, wrong_device);
  nvimgcodecInstance_t instance = nullptr;
  nvimgcodecExtension_t extension = nullptr;
  nvimgcodecEncoder_t encoder = nullptr;
  std::vector<nvimgcodecCodeStream_t> code_streams(3, nullptr);
  std::vector<nvimgcodecImage_t> images(3, nullptr);
  std::vector<std::vector<unsigned char>> in_buffers(3, std::vector<unsigned char>(3, 0x7f));
  std::vector<std::vector<unsigned char>> encoded_buffers(3);

  nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, extension_factory.getExtensionDesc()));

  nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
  exec_params.device_id = target_device;
  exec_params.max_num_cpu_threads = 2;
  exec_params.executor = executor.getDesc();
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr));

  auto resize_output = [](void* ctx, size_t bytes) -> unsigned char* {
    auto* buffer = reinterpret_cast<std::vector<unsigned char>*>(ctx);
    buffer->resize(bytes);
    return buffer->data();
  };

  for (size_t i = 0; i < images.size(); i++) {
    auto image_info = MakeHostRgbImageInfo(in_buffers[i].data());
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &images[i], &image_info));
    nvimgcodecImageInfo_t out_image_info = image_info;
    out_image_info.buffer = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance, &code_streams[i], &encoded_buffers[i], resize_output, &out_image_info));
  }

  nvimgcodecEncodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr};
  nvimgcodecFuture_t future = nullptr;
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder, images.data(), code_streams.data(), images.size(), &params, &future));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future));
  executor.wait();
  std::vector<nvimgcodecProcessingStatus_t> statuses(images.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
  size_t statuses_size = 0;
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &statuses_size));
  ASSERT_EQ(images.size(), statuses_size);
  for (auto status : statuses)
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
  EXPECT_TRUE(extension_factory.encoder().EncodeObservedDevicesMatch());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future));

  for (auto image : images)
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
  for (auto code_stream : code_streams)
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
  ASSERT_EQ(wrong_device, CurrentCudaDevice());
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
  ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(DeviceGuard, CustomAllocatorsUseStreamDevice_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  constexpr int wrong_device = 0;
  constexpr int target_device = 1;
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaSetDevice(target_device));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  AllocatorDeviceRecorder recorder;
  nvimgcodecDeviceAllocator_t device_allocator{NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, sizeof(nvimgcodecDeviceAllocator_t), nullptr,
      AllocatorDeviceRecorder::device_malloc, AllocatorDeviceRecorder::device_free, &recorder, 0};
  nvimgcodecPinnedAllocator_t pinned_allocator{NVIMGCODEC_STRUCTURE_TYPE_PINNED_ALLOCATOR, sizeof(nvimgcodecPinnedAllocator_t), nullptr,
      AllocatorDeviceRecorder::pinned_malloc, AllocatorDeviceRecorder::pinned_free, &recorder, 0};

  {
    CHECK_CUDA(cudaSetDevice(wrong_device));
    DeviceBuffer device_buffer(&device_allocator);
    device_buffer.resize(16, stream);
    EXPECT_EQ(target_device, recorder.device_malloc_device);
    EXPECT_EQ(target_device, recorder.device_malloc_pointer_device);
    EXPECT_EQ(target_device, PointerDevice(device_buffer.data));
  }
  EXPECT_EQ(target_device, recorder.device_free_device);
  EXPECT_EQ(target_device, recorder.device_free_pointer_device);

  {
    CHECK_CUDA(cudaSetDevice(wrong_device));
    PinnedBuffer pinned_buffer(&pinned_allocator);
    pinned_buffer.resize(16, stream);
    EXPECT_EQ(target_device, recorder.pinned_malloc_device);
  }
  EXPECT_EQ(target_device, recorder.pinned_free_device);

  CHECK_CUDA(cudaSetDevice(target_device));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaSetDevice(wrong_device));
}

TEST(DeviceGuard, DeviceBufferReallocatesWhenStreamDeviceChanges_MultiGPU) {
  if (!HasMultiGpu()) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices.";
  }

  cudaStream_t stream0 = nullptr;
  cudaStream_t stream1 = nullptr;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
  CHECK_CUDA(cudaSetDevice(1));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

  {
    DeviceBuffer device_buffer;
    CHECK_CUDA(cudaSetDevice(0));
    device_buffer.resize(16, stream0);
    EXPECT_EQ(0, PointerDevice(device_buffer.data));
    EXPECT_EQ(16u, device_buffer.capacity);

    EXPECT_EQ(0, CurrentCudaDevice());
    device_buffer.resize(8, stream1);
    EXPECT_EQ(0, CurrentCudaDevice());
    EXPECT_EQ(1, PointerDevice(device_buffer.data));
    EXPECT_EQ(8u, device_buffer.size);
    EXPECT_EQ(8u, device_buffer.capacity);
  }

  CHECK_CUDA(cudaSetDevice(1));
  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaStreamDestroy(stream0));
}

TEST(DeviceGuard, DeviceBufferReusesAllocationWhenStreamDeviceUnchanged) {
  cudaStream_t stream0 = nullptr;
  cudaStream_t stream1 = nullptr;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

  FailingDeviceAllocator recorder;
  nvimgcodecDeviceAllocator_t allocator{NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, sizeof(nvimgcodecDeviceAllocator_t),
      nullptr, FailingDeviceAllocator::device_malloc, FailingDeviceAllocator::device_free, &recorder, 0};

  {
    DeviceBuffer device_buffer(&allocator);
    device_buffer.resize(16, stream0);
    void* old_data = device_buffer.data;

    recorder.fail_next_alloc = true;
    EXPECT_NO_THROW(device_buffer.resize(8, stream1));
    EXPECT_EQ(old_data, device_buffer.data);
    EXPECT_EQ(8u, device_buffer.size);
    EXPECT_EQ(16u, device_buffer.capacity);
    EXPECT_EQ(stream1, device_buffer.stream);
  }

  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream0));
}

TEST(DeviceGuard, DeviceBufferDoesNotUseForeignFreeForBuiltInAllocation) {
  cudaStream_t stream0 = nullptr;
  cudaStream_t stream1 = nullptr;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

  FreeOnlyDeviceAllocatorRecorder recorder;
  nvimgcodecDeviceAllocator_t allocator{NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, sizeof(nvimgcodecDeviceAllocator_t),
      nullptr, nullptr, FreeOnlyDeviceAllocatorRecorder::device_free, &recorder, 0};

  {
    DeviceBuffer device_buffer(&allocator);
    device_buffer.resize(16, stream0);
    device_buffer.resize(8, stream1);
  }
  EXPECT_FALSE(recorder.device_free_called);

  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream0));
}

TEST(DeviceGuard, DeviceBufferKeepsOldAllocationWhenCustomAllocatorFails) {
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  FailingDeviceAllocator recorder;
  nvimgcodecDeviceAllocator_t allocator{NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, sizeof(nvimgcodecDeviceAllocator_t),
      nullptr, FailingDeviceAllocator::device_malloc, FailingDeviceAllocator::device_free, &recorder, 0};

  {
    DeviceBuffer device_buffer(&allocator);
    device_buffer.resize(16, stream);
    void* old_data = device_buffer.data;

    recorder.fail_next_alloc = true;
    EXPECT_THROW(device_buffer.resize(32, stream), Exception);
    EXPECT_EQ(old_data, device_buffer.data);
    EXPECT_EQ(16u, device_buffer.size);
    EXPECT_EQ(16u, device_buffer.capacity);
  }

  CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(DeviceGuard, PinnedBufferKeepsOldAllocationWhenCustomAllocatorFails) {
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  FailingPinnedAllocator recorder;
  nvimgcodecPinnedAllocator_t allocator{NVIMGCODEC_STRUCTURE_TYPE_PINNED_ALLOCATOR, sizeof(nvimgcodecPinnedAllocator_t),
      nullptr, FailingPinnedAllocator::pinned_malloc, FailingPinnedAllocator::pinned_free, &recorder, 0};

  {
    PinnedBuffer pinned_buffer(&allocator);
    pinned_buffer.resize(16, stream);
    void* old_data = pinned_buffer.data;

    recorder.fail_next_alloc = true;
    EXPECT_THROW(pinned_buffer.resize(32, stream), Exception);
    EXPECT_EQ(old_data, pinned_buffer.data);
    EXPECT_EQ(16u, pinned_buffer.size);
    EXPECT_EQ(16u, pinned_buffer.capacity);
  }

  CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(DeviceGuard, CheckOutOfRange) {
  int ndevs = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndevs));
  EXPECT_THROW(DeviceGuard{ndevs}, std::out_of_range);
}

} //namespace test
} // namespace nvimgcodec
