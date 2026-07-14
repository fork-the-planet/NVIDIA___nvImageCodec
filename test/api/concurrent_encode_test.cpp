/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * Regression test for the per-submission encoder params race.
 *
 * Two batched encodes are submitted to a single multi-threaded encoder with
 * different quality params; both params structs stay alive across both drains.
 * Each frame's encoded bytes must match the byte-level reference for the
 * quality requested at submit time.
 *
 * Before the fix, a shared mutable params pointer on the encoder was
 * overwritten by the second submit before the first submit's workers had
 * finished reading it — so most submit-A frames came back encoded at
 * submit-B's quality.
 */

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#if defined(BUILD_OPENCV_EXT)
#  include <extensions/opencv/opencv_ext.h>
#endif
#include <parsers/jpeg.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

// Buffer that backs nvimgcodecCodeStreamCreateToHostMem via a resize callback.
struct MemBuf {
    std::vector<uint8_t> bytes;
    static unsigned char* Resize(void* ctx, size_t n) {
        auto* b = static_cast<MemBuf*>(ctx);
        b->bytes.resize(n);
        return b->bytes.data();
    }
};

using ExtGetter = nvimgcodecStatus_t (*)(nvimgcodecExtensionDesc_t*);

static void RegisterExt(nvimgcodecInstance_t inst, std::vector<nvimgcodecExtension_t>& out, ExtGetter fn)
{
    nvimgcodecExtensionDesc_t d{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(d), 0};
    fn(&d);
    nvimgcodecExtension_t e = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(inst, &e, &d));
    out.push_back(e);
}

class ConcurrentEncodeTest : public ::testing::Test {
  protected:
    // Image and batch dimensions are tuned to keep encoder workers busy long
    // enough that the second submit's curr_params_ overwrite reliably races
    // against still-running workers from the first submit. Smaller batches /
    // images finish so quickly the window collapses and the race hides.
    static constexpr uint32_t kW = 512;
    static constexpr uint32_t kH = 512;
    static constexpr int kBatch = 16;

    void SetUp() override {
#if !defined(BUILD_OPENCV_EXT)
        GTEST_SKIP() << "opencv extension (CPU JPEG encoder) not built; skipping test";
#endif
        nvimgcodecInstanceCreateInfo_t ci{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(ci), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &ci));
        ASSERT_NO_FATAL_FAILURE(RegisterExt(instance_, exts_, get_jpeg_parser_extension_desc));
#if defined(BUILD_OPENCV_EXT)
        ASSERT_NO_FATAL_FAILURE(RegisterExt(instance_, exts_, get_opencv_extension_desc));
#endif
        // Synthesize a deterministic RGB image (interleaved, 8-bit, 3 channels).
        pixels_.resize(static_cast<size_t>(kW) * kH * 3);
        std::mt19937 rng(0xC0FFEE);
        for (auto& b : pixels_) b = static_cast<uint8_t>(rng() & 0xFF);
    }

    void TearDown() override {
        if (!instance_) return;
        for (auto e : exts_) ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(e));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    nvimgcodecEncoder_t MakeEncoder(int num_threads) {
        nvimgcodecExecutionParams_t ep{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(ep), 0};
        ep.device_id           = NVIMGCODEC_DEVICE_CPU_ONLY;
        ep.max_num_cpu_threads = num_threads;
        nvimgcodecEncoder_t enc = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecEncoderCreate(instance_, &enc, &ep, nullptr));
        return enc;
    }

    // Fill image_info for a kW x kH interleaved RGB 8-bit image written into pixels_.
    void FillImageInfo(nvimgcodecImageInfo_t& info) {
        info = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(info), 0};
        info.sample_format              = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.color_spec                 = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling         = NVIMGCODEC_SAMPLING_444;
        info.num_planes                 = 1;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type  = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision    = 8;
        info.plane_info[0].row_stride   = kW * 3;
        info.plane_info[0].width        = kW;
        info.plane_info[0].height       = kH;
        info.buffer_kind                = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        info.buffer                     = pixels_.data();
        std::strcpy(info.codec_name, "jpeg");
    }

    // Encode the synthetic pixels once with the given encoder + params and return
    // the resulting JPEG bytes. Used to build per-quality references. Returns
    // an empty vector on any failure, releasing partially-created handles so
    // callers can branch on `.empty()` without worrying about leaked state.
    std::vector<uint8_t> EncodeOnce(nvimgcodecEncoder_t enc, const nvimgcodecEncodeParams_t& params) {
        nvimgcodecImageInfo_t img_info;
        FillImageInfo(img_info);
        nvimgcodecImage_t img = nullptr;
        auto status = nvimgcodecImageCreate(instance_, &img, &img_info);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, status);
        if (status != NVIMGCODEC_STATUS_SUCCESS) return {};

        MemBuf out;
        nvimgcodecImageInfo_t cs_info(img_info);
        nvimgcodecCodeStream_t cs = nullptr;
        status = nvimgcodecCodeStreamCreateToHostMem(instance_, &cs, &out, MemBuf::Resize, &cs_info);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, status);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecImageDestroy(img);
            return {};
        }

        nvimgcodecFuture_t fut = nullptr;
        status = nvimgcodecEncoderEncode(enc, &img, &cs, 1, &params, &fut);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, status);
        if (status != NVIMGCODEC_STATUS_SUCCESS || fut == nullptr) {
            nvimgcodecCodeStreamDestroy(cs);
            nvimgcodecImageDestroy(img);
            return {};
        }
        status = nvimgcodecFutureWaitForAll(fut);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, status);
        nvimgcodecProcessingStatus_t st = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t sz = 0;
        nvimgcodecFutureGetProcessingStatus(fut, &st, &sz);
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, st);
        nvimgcodecFutureDestroy(fut);

        nvimgcodecCodeStreamDestroy(cs);
        nvimgcodecImageDestroy(img);
        if (status != NVIMGCODEC_STATUS_SUCCESS || st != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
            return {};
        return out.bytes;
    }

    nvimgcodecInstance_t instance_ = nullptr;
    std::vector<nvimgcodecExtension_t> exts_;
    std::vector<uint8_t> pixels_;
};

TEST_F(ConcurrentEncodeTest, back_to_back_different_quality_no_stomp)
{
    // Two distinct quality params.
    nvimgcodecEncodeParams_t params_a{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(params_a), nullptr};
    params_a.quality_type  = NVIMGCODEC_QUALITY_TYPE_QUALITY;
    params_a.quality_value = 10.0f;

    nvimgcodecEncodeParams_t params_b{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(params_b), nullptr};
    params_b.quality_type  = NVIMGCODEC_QUALITY_TYPE_QUALITY;
    params_b.quality_value = 95.0f;

    // Build references on a single-threaded encoder (single-threaded path is
    // unaffected by the race). Compare ENCODED SIZES rather than bytes:
    // libjpeg/opencv produce deterministic bytes for a given input + quality,
    // but byte equality is brittle if opencv changes SIMD path or quantization
    // rounding. Size is a strong-and-robust discriminator — q=10 produces a
    // file ~20× smaller than q=95 on 512×512 random RGB, so a stomp surfaces
    // as an order-of-magnitude size mismatch.
    nvimgcodecEncoder_t ref_enc = MakeEncoder(/*num_threads=*/1);
    auto ref_q10 = EncodeOnce(ref_enc, params_a);
    auto ref_q95 = EncodeOnce(ref_enc, params_b);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(ref_enc));
    ASSERT_FALSE(ref_q10.empty()) << "reference encode at q=10 produced no bytes";
    ASSERT_FALSE(ref_q95.empty()) << "reference encode at q=95 produced no bytes";
    ASSERT_LT(ref_q10.size() * 5, ref_q95.size())
        << "references must differ in size by ~5x or more — test setup is broken "
        << "(ref_q10=" << ref_q10.size() << " ref_q95=" << ref_q95.size() << ")";

    // Multi-threaded encoder forces workers to read per-sample params
    // asynchronously — the regression path covered by this test.
    nvimgcodecEncoder_t mt_enc = MakeEncoder(/*num_threads=*/8);

    // Prepare two batches.
    struct Batch {
        std::vector<MemBuf> outs;
        std::vector<nvimgcodecImage_t> images;
        std::vector<nvimgcodecCodeStream_t> streams;
        nvimgcodecFuture_t future = nullptr;
    };
    Batch a, b;
    a.outs.resize(kBatch); a.images.resize(kBatch, nullptr); a.streams.resize(kBatch, nullptr);
    b.outs.resize(kBatch); b.images.resize(kBatch, nullptr); b.streams.resize(kBatch, nullptr);

    auto destroy_batch = [&](Batch& bat) {
        for (int i = 0; i < kBatch; ++i) {
            if (bat.streams[i]) nvimgcodecCodeStreamDestroy(bat.streams[i]);
            if (bat.images[i])  nvimgcodecImageDestroy(bat.images[i]);
            bat.streams[i] = nullptr;
            bat.images[i]  = nullptr;
        }
    };
    auto setup_batch = [&](Batch& bat) {
        for (int i = 0; i < kBatch; ++i) {
            nvimgcodecImageInfo_t img_info;
            FillImageInfo(img_info);
            if (nvimgcodecImageCreate(instance_, &bat.images[i], &img_info) != NVIMGCODEC_STATUS_SUCCESS) {
                destroy_batch(bat);
                FAIL() << "nvimgcodecImageCreate failed for frame " << i;
            }
            nvimgcodecImageInfo_t cs_info(img_info);
            if (nvimgcodecCodeStreamCreateToHostMem(instance_, &bat.streams[i],
                    &bat.outs[i], MemBuf::Resize, &cs_info) != NVIMGCODEC_STATUS_SUCCESS) {
                destroy_batch(bat);
                FAIL() << "nvimgcodecCodeStreamCreateToHostMem failed for frame " << i;
            }
        }
    };
    ASSERT_NO_FATAL_FAILURE(setup_batch(a));
    ASSERT_NO_FATAL_FAILURE(setup_batch(b));

    // Back-to-back submits. params_a and params_b are kept alive past both drains.
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecEncoderEncode(mt_enc, a.images.data(), a.streams.data(),
            kBatch, &params_a, &a.future));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecEncoderEncode(mt_enc, b.images.data(), b.streams.data(),
            kBatch, &params_b, &b.future));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(a.future));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(b.future));

    std::vector<nvimgcodecProcessingStatus_t> sa(kBatch, NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
    std::vector<nvimgcodecProcessingStatus_t> sb(kBatch, NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
    size_t na = 0, nb = 0;
    nvimgcodecFutureGetProcessingStatus(a.future, sa.data(), &na);
    nvimgcodecFutureGetProcessingStatus(b.future, sb.data(), &nb);
    nvimgcodecFutureDestroy(a.future);
    nvimgcodecFutureDestroy(b.future);
    ASSERT_EQ(static_cast<size_t>(kBatch), na);
    ASSERT_EQ(static_cast<size_t>(kBatch), nb);

    // A frame whose encoded size is closer to the *other* submission's
    // reference size than to its own was encoded with the wrong params — the
    // race signature.
    auto closer_to_first = [](size_t actual, size_t ref_own, size_t ref_other) {
        auto d_own   = actual > ref_own   ? actual - ref_own   : ref_own   - actual;
        auto d_other = actual > ref_other ? actual - ref_other : ref_other - actual;
        return d_own < d_other;
    };

    for (int i = 0; i < kBatch; ++i) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, sa[i]) << "submit A frame " << i;
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, sb[i]) << "submit B frame " << i;
        EXPECT_TRUE(closer_to_first(a.outs[i].bytes.size(), ref_q10.size(), ref_q95.size()))
            << "submit A frame " << i << " size " << a.outs[i].bytes.size()
            << " closer to q=95 reference (" << ref_q95.size()
            << ") than q=10 (" << ref_q10.size() << ") — encoded at wrong quality";
        EXPECT_TRUE(closer_to_first(b.outs[i].bytes.size(), ref_q95.size(), ref_q10.size()))
            << "submit B frame " << i << " size " << b.outs[i].bytes.size()
            << " closer to q=10 reference (" << ref_q10.size()
            << ") than q=95 (" << ref_q95.size() << ") — encoded at wrong quality";
    }

    for (int i = 0; i < kBatch; ++i) {
        nvimgcodecCodeStreamDestroy(a.streams[i]);
        nvimgcodecCodeStreamDestroy(b.streams[i]);
        nvimgcodecImageDestroy(a.images[i]);
        nvimgcodecImageDestroy(b.images[i]);
    }
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(mt_enc));
}

}} // namespace nvimgcodec::test
