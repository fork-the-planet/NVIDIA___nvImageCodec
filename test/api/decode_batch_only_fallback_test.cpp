/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstring>
#include <deque>
#include <vector>

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "parsers/parser_test_utils.h"

namespace nvimgcodec { namespace test {

namespace {

// Minimal 1x1 24-bit BMP, recognized by the builtin bmp parser.
static unsigned char small_bmp[] = {0x42, 0x4D, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00};

// A decoder whose descriptor mirrors the nvjpeg lossless decoder: it implements ONLY the
// batched API (decodeBatch) and leaves the single-sample `decode` callback NULL. Its
// decodeBatch always reports failure, to drive the framework down the decode-failure
// fallback path.
class BatchOnlyFailingDecoder
{
  public:
    BatchOnlyFailingDecoder(const nvimgcodecFrameworkDesc_t* /*framework*/, const char* id)
        : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), nullptr,
              this, id, "bmp", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
              static_create, static_destroy, static_get_metadata, static_can_decode,
              nullptr,             // decode (single-sample) intentionally NULL, like nvjpeg lossless
              static_decode_batch, // only the batched API is implemented
              nullptr}
    {
    }
    nvimgcodecDecoderDesc_t* getDecoderDesc() { return &decoder_desc_; }

  private:
    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t*, const char*)
    {
        *decoder = static_cast<nvimgcodecDecoder_t>(instance);
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t) { return NVIMGCODEC_STATUS_SUCCESS; }
    static nvimgcodecStatus_t static_get_metadata(
        nvimgcodecDecoder_t, const nvimgcodecCodeStreamDesc_t*, nvimgcodecMetadata_t**, int*)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t, const nvimgcodecImageDesc_t*,
        const nvimgcodecCodeStreamDesc_t*, const nvimgcodecDecodeParams_t*, int)
    {
        return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    }
    // Mimics nvjpegDecodeBatched returning an execution error.
    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t, const nvimgcodecImageDesc_t** images,
        const nvimgcodecCodeStreamDesc_t**, int batch_size, const nvimgcodecDecodeParams_t*, int)
    {
        for (int i = 0; i < batch_size; ++i)
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    nvimgcodecDecoderDesc_t decoder_desc_;
};

// Registers two batch-only decoders for the "bmp" codec. The higher-priority one's
// decodeBatch fails first; the framework then falls back to the second, which is also
// batch-only (decode == NULL). The decode-failure fallback must not invoke a NULL
// single-sample `decode` callback.
struct TwoBatchOnlyDecodersExtension
{
    TwoBatchOnlyDecodersExtension()
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, this,
              "two_batch_only_decoders_ext", NVIMGCODEC_VER, NVIMGCODEC_VER, static_extension_create, static_extension_destroy}
    {
    }
    nvimgcodecExtensionDesc_t* getExtensionDesc() { return &desc_; }

    struct Extension
    {
        explicit Extension(const nvimgcodecFrameworkDesc_t* framework)
            : framework_(framework)
        {
            decoders_.emplace_back(framework, "batch_only_a");
            decoders_.emplace_back(framework, "batch_only_b");
            // batch_only_a is tried first; batch_only_b is its fallback (both batch-only).
            framework->registerDecoder(framework->instance, decoders_[0].getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
            framework->registerDecoder(framework->instance, decoders_[1].getDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL);
        }
        ~Extension()
        {
            for (auto& d : decoders_)
                framework_->unregisterDecoder(framework_->instance, d.getDecoderDesc());
        }
        const nvimgcodecFrameworkDesc_t* framework_;
        std::deque<BatchOnlyFailingDecoder> decoders_; // deque keeps descriptor addresses stable
    };

    static nvimgcodecStatus_t static_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new Extension(framework));
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    static nvimgcodecStatus_t static_extension_destroy(nvimgcodecExtension_t extension)
    {
        delete reinterpret_cast<Extension*>(extension);
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    nvimgcodecExtensionDesc_t desc_;
};

} // namespace

// Regression test: a decoder that implements only the batched API (decode == NULL, like the
// nvjpeg lossless decoder) must not cause a NULL function-pointer call when a decodeBatch
// failure routes the sample down the single-sample fallback path. Before the fix this
// segfaulted (call to 0x0 from ImageDecoder::decode); the guard was only an assert(), which is
// stripped in release builds.
TEST(NvImageCodecsDecodeBatchOnlyFallbackTest, BatchOnlyDecoderFailureDoesNotCallNullDecode)
{
    nvimgcodecInstanceCreateInfo_t create_info{
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    create_info.load_builtin_modules = 1; // builtin bmp parser
    nvimgcodecInstance_t instance = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));

    TwoBatchOnlyDecodersExtension ext_factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, ext_factory.getExtensionDesc()));

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
    exec_params.max_num_cpu_threads = 1;
    nvimgcodecDecoder_t decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    nvimgcodecCodeStream_t stream = nullptr;
    LoadImageFromHostMemory(instance, stream, small_bmp, sizeof(small_bmp));

    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream, &image_info));
    size_t buffer_size = 0;
    for (uint32_t p = 0; p < image_info.num_planes; ++p)
        buffer_size += static_cast<size_t>(image_info.plane_info[p].row_stride) * image_info.plane_info[p].height;
    if (buffer_size == 0)
        buffer_size = 64;
    std::vector<uint8_t> out_buffer(buffer_size, 0);
    image_info.buffer = out_buffer.data();
    image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    nvimgcodecImage_t image = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &image, &image_info));

    nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    nvimgcodecFuture_t future = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder, &stream, &image, 1, &params, &future));
    // Before the fix, the worker thread segfaults here (NULL `decode` callback).
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future));

    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    size_t status_size = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future, &status, &status_size));
    EXPECT_EQ(1u, status_size);
    // Every (mock) decoder fails, so the sample must report failure — gracefully, not by crashing.
    EXPECT_NE(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

    nvimgcodecFutureDestroy(future);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

}} // namespace nvimgcodec::test
