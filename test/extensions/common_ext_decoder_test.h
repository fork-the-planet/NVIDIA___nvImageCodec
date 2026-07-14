/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include "imgproc/type_utils.h"
#include <test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <type_traits>
#include <vector>
#include <tuple>
#include "nvimgcodec_tests.h"

#define DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcodec { namespace test {

class CommonExtDecoderTest : public ::testing::Test
{
  public:
    // Controls the row_stride of the output planes the decoder writes into, so
    // the per-plane stride handling of each codec can be exercised:
    //   Contiguous       - row_stride == natural row size (no padding)
    //   SamePadAllPlanes - every plane gets the same extra row padding
    //   DiffPadPerPlane  - each plane gets a distinct extra row padding
    enum class StrideMode { Contiguous, SamePadAllPlanes, DiffPadPerPlane };

    // Extra bytes appended to plane p's natural row size for the given mode.
    static size_t ExtraRowPadding(StrideMode mode, uint32_t plane)
    {
        switch (mode) {
        case StrideMode::Contiguous:       return 0;
        case StrideMode::SamePadAllPlanes: return 32;
        case StrideMode::DiffPadPerPlane:  return 16 * (plane + 1);  // 16, 32, 48, ...
        }
        return 0;
    }

    static const char* StrideModeName(StrideMode mode)
    {
        switch (mode) {
        case StrideMode::Contiguous:       return "Contiguous";
        case StrideMode::SamePadAllPlanes: return "SamePadAllPlanes";
        case StrideMode::DiffPadPerPlane:  return "DiffPadPerPlane";
        }
        return "?";
    }

    void SetUp()
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        images_.clear();
        streams_.clear();

        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params_.apply_exif_orientation= 1;
    }

    void CreateDecoder()
    {
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params_.apply_exif_orientation = 1;
    }

    void TearDown()
    {
        if (decoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        }
        if (future_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future_));
        }
        if (image_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image_));
        }
        if (in_code_stream_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(in_code_stream_));
        }
        for (auto& ext : extensions_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(ext));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    // Templated core of the reference-image decode test. T is the C++ element
    // type of the decoded output and the reference (uint8_t / uint16_t / float).
    //   sample_type   - nvimgcodec sample type matching T
    //   cv_base_depth - OpenCV depth matching T (CV_8U / CV_16U / CV_32F)
    //   ref_ext       - reference file extension (".ppm" for 8/16-bit, ".pfm" for fp32)
    //   eps           - per-element absolute tolerance (0 for the lossless high-precision paths)
    // The reference is loaded with OpenCV and compared against nvimagecodec's
    // decode honouring the same planar / BGR / ROI / per-plane-stride handling
    // for every element type.
    template <typename T>
    void TestSingleImageImpl(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format,
        nvimgcodecSampleDataType_t sample_type, int cv_base_depth, const std::string& ref_ext, double eps,
        nvimgcodecRegion_t region, StrideMode stride_mode)
    {
        constexpr bool kHighPrec = !std::is_same<T, uint8_t>::value;

        // Allow repeated invocations within one test (e.g. one per stride mode):
        // release the per-call handles from any previous call so they don't leak. Image_ and CodeStream_ can be reused
        if (future_) { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future_)); future_ = nullptr; }

        std::string filename = resources_dir + "/" + rel_path;
        std::string reference_filename = std::filesystem::path(resources_dir + "/ref/" + rel_path).replace_extension(ref_ext).string();

        int num_channels = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y ? 1 : 3;
        // check if we are decoding grayscale image and we don't change channels:
        if (sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED || sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
            if (reference_filename.ends_with("grayscale.ppm")) {
                num_channels = 1;
            }
        }

        int cv_type = CV_MAKETYPE(cv_base_depth, num_channels);
        // For higher-than-8-bit references IMREAD_UNCHANGED preserves the native
        // depth and channel count; 8-bit keeps the original COLOR/GRAYSCALE flags.
        int cv_flags = kHighPrec ? cv::IMREAD_UNCHANGED
                                 : (num_channels == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
        cv::Mat ref;
        if (region.ndim == 0) {
            ref = cv::imread(reference_filename, cv_flags);
        } else {
            int start_x = region.start[1];
            int start_y = region.start[0];
            int crop_w = region.end[1] - region.start[1];
            int crop_h = region.end[0] - region.start[0];
            cv::Mat tmp = cv::imread(reference_filename, cv_flags);
            cv::Rect roi(start_x, start_y, crop_w, crop_h);
            tmp(roi).copyTo(ref);
        }
        ASSERT_FALSE(ref.empty()) << "Failed to load reference " << reference_filename;

        bool planar = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB || sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR ||
                      sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y || sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
        bool bgr = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR || sample_format == NVIMGCODEC_SAMPLEFORMAT_I_BGR;
        if (!bgr && num_channels >= 3)
            ref = bgr2rgb(ref);

        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        // Set chroma_subsampling based on number of channels being decoded
        image_info_.chroma_subsampling = (num_channels < 3) ? NVIMGCODEC_SAMPLING_GRAY : NVIMGCODEC_SAMPLING_444;
        uint32_t& width = image_info_.plane_info[0].width;
        uint32_t& height = image_info_.plane_info[0].height;

        bool swap_xy = params_.apply_exif_orientation && image_info_.orientation.rotated % 180 == 90;
        if (swap_xy) {
            std::swap(width, height);
        }

        nvimgcodecCodeStream_t sub_code_stream = nullptr;

        if (region.ndim != 0) {
            assert(region.ndim == 2);
            width = region.end[1] - region.start[1];
            height = region.end[0] - region.start[0];
            nvimgcodecCodeStreamView_t code_stream_view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr};
            code_stream_view.image_idx = 0;
            code_stream_view.region = region;

            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetSubCodeStream(in_code_stream_, &sub_code_stream, &code_stream_view));
        }

        image_info_.num_planes = planar ? num_channels : 1;
        int plane_nchannels = planar ? 1 : num_channels;
        for (uint32_t p = 0; p < image_info_.num_planes; p++) {
            image_info_.plane_info[p].width = width;
            image_info_.plane_info[p].height = height;
            // Natural (contiguous) row size in bytes, optionally padded per the
            // stride mode so each plane may carry its own row_stride.
            image_info_.plane_info[p].row_stride = width * plane_nchannels * sizeof(T) + ExtraRowPadding(stride_mode, p);
            image_info_.plane_info[p].num_channels = plane_nchannels;
            image_info_.plane_info[p].sample_type = sample_type;
            image_info_.plane_info[p].precision = 0; // Compute precision based on sample_type.
        }
        // GetBufferSize sums row_stride * height per plane, so it accounts for
        // any per-plane row padding applied above.
        out_buffer_.resize(GetBufferSize(image_info_));
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
        auto* cs = region.ndim != 0 ? &sub_code_stream : &in_code_stream_;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, cs, &image_, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        nvimgcodecProcessingStatus_t status;
        size_t status_size;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
        ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);

        if (region.ndim != 0) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(sub_code_stream));
        }

        ASSERT_EQ(ref.size[0], height);
        ASSERT_EQ(ref.size[1], width);
        ASSERT_EQ(ref.type(), cv_type);
#if DEBUG_DUMP_DECODE_OUTPUT
        cv::Mat decoded_image(height, width, cv_type, static_cast<void*>(out_buffer_.data()));
        cv::imwrite("./decode_out.ppm", rgb2bgr(decoded_image));
        cv::imwrite("./ref.ppm", rgb2bgr(ref));
#endif

        // Byte offset of the start of each plane (running sum of
        // row_stride * height), so the read honours per-plane padding.
        size_t plane_offset[NVIMGCODEC_MAX_NUM_PLANES] = {0};
        {
            size_t running = 0;
            for (uint32_t p = 0; p < image_info_.num_planes; ++p) {
                plane_offset[p] = running;
                running += image_info_.plane_info[p].row_stride * image_info_.plane_info[p].height;
            }
        }
        if (planar) {
            for (int c = 0; c < num_channels; c++) {
                size_t row_stride = image_info_.plane_info[c].row_stride;
                for (size_t i = 0; i < height; i++) {
                    const T* out_row = reinterpret_cast<const T*>(out_buffer_.data() + plane_offset[c] + i * row_stride);
                    const T* ref_row = ref.ptr<T>(i);
                    for (size_t j = 0; j < width; j++) {
                        T out_val = out_row[j];
                        T ref_val = ref_row[j * num_channels + c];
                        // Unary + integer-promotes uint8/uint16 to int (so they print as
                        // numbers, not characters) while leaving float values intact.
                        ASSERT_NEAR(out_val, ref_val, eps)
                            << "@" << i << "x" << j << "x" << c << " : " << +out_val << " != " << +ref_val << "\n";
                    }
                }
            }
        } else {
            size_t row_stride = image_info_.plane_info[0].row_stride;
            for (size_t i = 0; i < height; i++) {
                const T* out_row = reinterpret_cast<const T*>(out_buffer_.data() + i * row_stride);
                const T* ref_row = ref.ptr<T>(i);
                for (size_t j = 0; j < width; j++) {
                    for (int c = 0; c < num_channels; c++) {
                        T out_val = out_row[j * num_channels + c];
                        T ref_val = ref_row[j * num_channels + c];
                        // Unary + integer-promotes uint8/uint16 to int (so they print as
                        // numbers, not characters) while leaving float values intact.
                        ASSERT_NEAR(out_val, ref_val, eps)
                            << "@" << i << "x" << j << "x" << c << " : " << +out_val << " != " << +ref_val << "\n";
                    }
                }
            }
        }
    }

    // uint8 reference-image test (unchanged behaviour): 8-bit .ppm reference,
    // with the legacy per-image epsilon heuristics.
    void TestSingleImage(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format,
        nvimgcodecRegion_t region = {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0},
        StrideMode stride_mode = StrideMode::Contiguous)
    {
        double eps = 1;
        if (rel_path.find("exif") != std::string::npos) {
            eps = 4;
        }
        else if (rel_path.find("cmyk") != std::string::npos) {
            eps = 2;
        }
        else if (rel_path.find("ycbcr") != std::string::npos) {
            eps = 4;
        }
        TestSingleImageImpl<uint8_t>(rel_path, sample_format, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, CV_8U, ".ppm", eps,
            region, stride_mode);
    }

    // Convenience overload: decode the whole image (no ROI) with the given
    // stride mode. Used by the per-mode SingleImage test cases.
    void TestSingleImage(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format, StrideMode stride_mode)
    {
        TestSingleImage(rel_path, sample_format,
            {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0}, stride_mode);
    }

    // Higher-precision reference-image test. The reference is nvimagecodec's own
    // native-depth decode (frozen, OpenCV-verified at generation time), so the
    // comparison is bit-exact (eps = 0). Dispatches on the requested sample type:
    //   UINT16  -> 16-bit .ppm reference
    //   FLOAT32 -> .pfm reference
    void TestSingleImageHighPrec(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format,
        nvimgcodecSampleDataType_t sample_type, StrideMode stride_mode = StrideMode::Contiguous)
    {
        nvimgcodecRegion_t no_region{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0};
        switch (sample_type) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
            TestSingleImageImpl<uint16_t>(rel_path, sample_format, sample_type, CV_16U, ".ppm", 0.0, no_region, stride_mode);
            break;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
            TestSingleImageImpl<float>(rel_path, sample_format, sample_type, CV_32F, ".pfm", 0.0, no_region, stride_mode);
            break;
        default:
            FAIL() << "Unsupported high-precision sample type: " << sample_type;
        }
    }

    void TestNotSupported(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format, nvimgcodecSampleDataType_t sample_type,
        nvimgcodecProcessingStatus_t expected_status)
    {
        std::string filename = resources_dir + "/" + rel_path;

        int num_channels = sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y ? 1 : 3;
        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width * num_channels;
        image_info_.plane_info[0].num_channels = num_channels;
        image_info_.plane_info[0].sample_type = sample_type;
        out_buffer_.resize(GetBufferSize(image_info_));
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        nvimgcodecProcessingStatus_t status;
        size_t status_size;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
        ASSERT_EQ(expected_status, status);
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecodeParams_t params_;
    nvimgcodecImageInfo_t image_info_;
    nvimgcodecCodeStream_t in_code_stream_ = nullptr;
    nvimgcodecImage_t image_ = nullptr;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<nvimgcodecCodeStream_t> streams_;
    nvimgcodecFuture_t future_ = nullptr;
    std::vector<uint8_t> out_buffer_;
    std::vector<nvimgcodecExtension_t> extensions_;

};

class CommonExtDecoderTestWithPathAndFormat : 
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleFormat_t>>,
    public CommonExtDecoderTest 
{
public:
    void SetUp() override
    {
        image_path = std::get<0>(GetParam());
        sample_format = std::get<1>(GetParam());
        CommonExtDecoderTest::SetUp();
    }

    std::string image_path;
    nvimgcodecSampleFormat_t sample_format;
};

// Emit three SingleImage test cases for a path+format decoder fixture, one per
// plane stride layout, so each layout is reported as a distinct gtest case.
// The fixture must expose `image_path` and `sample_format` members.
#define DEFINE_SINGLE_IMAGE_STRIDE_TESTS(FIXTURE)                                                 \
    TEST_P(FIXTURE, SingleImageStrideContiguous)                                                  \
    {                                                                                             \
        TestSingleImage(image_path, sample_format, StrideMode::Contiguous);                       \
    }                                                                                             \
    TEST_P(FIXTURE, SingleImageStrideSamePadAllPlanes)                                            \
    {                                                                                             \
        TestSingleImage(image_path, sample_format, StrideMode::SamePadAllPlanes);                 \
    }                                                                                             \
    TEST_P(FIXTURE, SingleImageStrideDiffPadPerPlane)                                             \
    {                                                                                             \
        TestSingleImage(image_path, sample_format, StrideMode::DiffPadPerPlane);                  \
    }

// Higher-precision counterpart of CommonExtDecoderTestWithPathAndFormat: the
// parameter tuple additionally carries the output sample type so a single
// fixture can drive uint16 / fp32 reference tests.
class CommonExtDecoderTestWithPathFormatAndType :
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleFormat_t, nvimgcodecSampleDataType_t>>,
    public CommonExtDecoderTest
{
public:
    void SetUp() override
    {
        image_path = std::get<0>(GetParam());
        sample_format = std::get<1>(GetParam());
        sample_type = std::get<2>(GetParam());
        CommonExtDecoderTest::SetUp();
    }

    std::string image_path;
    nvimgcodecSampleFormat_t sample_format;
    nvimgcodecSampleDataType_t sample_type;
};

// Emit three high-precision SingleImage test cases for a path+format+type
// decoder fixture, one per plane stride layout. The fixture must expose
// `image_path`, `sample_format` and `sample_type` members.
#define DEFINE_SINGLE_IMAGE_HIGHPREC_STRIDE_TESTS(FIXTURE)                                        \
    TEST_P(FIXTURE, HighPrecSingleImageStrideContiguous)                                          \
    {                                                                                             \
        TestSingleImageHighPrec(image_path, sample_format, sample_type, StrideMode::Contiguous);  \
    }                                                                                             \
    TEST_P(FIXTURE, HighPrecSingleImageStrideSamePadAllPlanes)                                    \
    {                                                                                             \
        TestSingleImageHighPrec(image_path, sample_format, sample_type, StrideMode::SamePadAllPlanes); \
    }                                                                                             \
    TEST_P(FIXTURE, HighPrecSingleImageStrideDiffPadPerPlane)                                     \
    {                                                                                             \
        TestSingleImageHighPrec(image_path, sample_format, sample_type, StrideMode::DiffPadPerPlane); \
    }

}} // namespace nvimgcodec::test
