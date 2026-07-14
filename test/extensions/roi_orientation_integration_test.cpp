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

// Integration test for the documented coordinate-system contract of nvimgcodecRegion_t:
// when apply_exif_orientation is set, the region is interpreted in *display* coordinates,
// so a region that exceeds the *raw codestream* dimensions but fits in the post-EXIF
// (display) dimensions must be accepted and decoded correctly.
//
// padlock-406986_640_rotate_90.jpg encodes an image whose JPEG SOF dimensions are 640
// wide x 426 tall (raw). EXIF orientation 6 (ROTATE_90_CW) means the displayed image is
// 426 wide x 640 tall. Region y in [0, 500) x in [0, 200) is OOB in raw codestream
// coords (raw_h=426 < 500) but fits in display coords (display_h=640). Pre-fix, the
// decoders' canDecode flagged this region as OOB against raw dims and the framework
// could not find any decoder to handle it.
//
// We parametrise over a representative backend per decoder so this guard covers:
//   - opencv (CPU_ONLY)            — bounds check + post-decode crop in display coords
//   - nvjpeg cuda (HYBRID_CPU_GPU) — bounds check + nvjpegDecodeParamsSetROI in display coords
//   - nvjpeg HW (HW_GPU_ONLY)      — same; skipped at runtime if not present
// Backends that can't be created on the host (no HW decoder, etc.) are skipped.

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <test_utils.h>
#include <parsers/jpeg.h>
#include <parsers/parser_test_utils.h>
#include <extensions/opencv/opencv_ext.h>
#include <extensions/libjpeg_turbo/libjpeg_turbo_ext.h>
#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

// Each case loads the minimal set of decoder extensions that route to the named
// decoder for an EXIF-rotated JPEG. Important: we do NOT load fallback decoders here,
// because the framework will silently fall back to e.g. opencv when nvjpeg refuses,
// and that would mask a regression in nvjpeg's own canDecode path. Each row exercises
// exactly one decoder.
struct BackendCase {
    const char* name;
    nvimgcodecBackendKind_t kind;
    // Which decoder extensions to load. The JPEG parser is always loaded.
    bool with_libjpeg_turbo;
    bool with_opencv;
    bool with_nvjpeg;
    // Whether the loaded decoder set is expected to apply EXIF orientation. libjpeg_turbo
    // refuses orientation outright, so when it is the only decoder in play the
    // display-coord ROI case can't decode and we skip that sub-test for this backend.
    bool can_apply_orientation;
};

constexpr BackendCase kBackendCases[] = {
    // libjpeg_turbo only — exercises libjpeg-turbo's partial-decode ROI path.
    {"libjpeg_turbo", NVIMGCODEC_BACKEND_KIND_CPU_ONLY,       /*ljt=*/true,  /*ocv=*/false, /*nvj=*/false, /*orient=*/false},
    // opencv only — exercises opencv's full-decode-then-crop path with orientation.
    {"opencv",        NVIMGCODEC_BACKEND_KIND_CPU_ONLY,       /*ljt=*/false, /*ocv=*/true,  /*nvj=*/false, /*orient=*/true},
    // HYBRID_CPU_GPU: nvjpeg cuda decoder only — no fallback.
    {"nvjpeg_cuda",   NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, /*ljt=*/false, /*ocv=*/false, /*nvj=*/true,  /*orient=*/true},
    // HW_GPU_ONLY: nvjpeg HW decoder only. Skipped at runtime if no HW JPEG decoder.
    {"nvjpeg_hw",     NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,    /*ljt=*/false, /*ocv=*/false, /*nvj=*/true,  /*orient=*/true},
};

class RoiOrientationIntegrationTest : public ::testing::TestWithParam<BackendCase>
{
  public:
    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        nvimgcodecExtensionDesc_t jpeg_parser_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg_parser_desc));

        if (GetParam().with_libjpeg_turbo) {
            nvimgcodecExtensionDesc_t libjpeg_turbo_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_libjpeg_turbo_extension_desc(&libjpeg_turbo_desc));
            extensions_.emplace_back();
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &libjpeg_turbo_desc));
        }

        if (GetParam().with_opencv) {
            nvimgcodecExtensionDesc_t opencv_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_opencv_extension_desc(&opencv_desc));
            extensions_.emplace_back();
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &opencv_desc));
        }

        if (GetParam().with_nvjpeg) {
            nvimgcodecExtensionDesc_t nvjpeg_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvjpeg_extension_desc(&nvjpeg_desc));
            extensions_.emplace_back();
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvjpeg_desc));
        }

        // Restrict the decoder to the parametrised backend. If the create fails for this
        // backend (e.g. HW decoder unavailable on the host) skip the test.
        backend_ = nvimgcodecBackend_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr,
                                       GetParam().kind, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f}};
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;
        exec_params.num_backends = 1;
        exec_params.backends = &backend_;
        // Disable fancy chroma upsampling so nvjpeg's ROI mode and full decode produce
        // identical pixels at MCU boundaries; without this the test would need a loose
        // pixel tolerance that could mask real coordinate-system regressions.
        // ROI upsampling was fixed in nvJPEG 13.2 (available in CTK 13.3) but we don't have a good way to detect
        // nvJPEG version here, so we just disable fancy upsampling for all nvJPEG versions
        auto st = nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, ":fancy_upsampling=0");
        if (st != NVIMGCODEC_STATUS_SUCCESS) {
            decoder_ = nullptr;
            GTEST_SKIP() << "Backend " << GetParam().name << " unavailable (decoder create returned " << st << ")";
        }
    }

    void TearDown() override
    {
        if (image_) { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image_)); }
        if (future_) { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future_)); }
        if (sub_code_stream_) { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(sub_code_stream_)); }
        if (code_stream_) { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream_)); }
        if (decoder_) { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_)); }
        for (auto& ext : extensions_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(ext));
        }
        if (device_buffer_) {
            ASSERT_EQ(cudaSuccess, cudaFree(device_buffer_));
            device_buffer_ = nullptr;
        }
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    // Load image, set up I_RGB output of correct buffer kind for the backend.
    void PrepareImage(const std::string& abs_path, int32_t out_height, int32_t out_width,
                      const nvimgcodecRegion_t* region)
    {
        LoadImageFromFilename(instance_, code_stream_, abs_path);

        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(code_stream_, &info));
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        info.num_planes = 1;
        info.plane_info[0].width = out_width;
        info.plane_info[0].height = out_height;
        info.plane_info[0].row_stride = static_cast<size_t>(out_width) * 3;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 0;

        if (region != nullptr) {
            nvimgcodecCodeStreamView_t view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr};
            view.image_idx = 0;
            view.region = *region;
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetSubCodeStream(code_stream_, &sub_code_stream_, &view));
        }

        const size_t out_bytes = static_cast<size_t>(out_height) * out_width * 3;
        if (GetParam().kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY) {
            host_buffer_.assign(out_bytes, 0);
            info.buffer = host_buffer_.data();
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        } else {
            ASSERT_EQ(cudaSuccess, cudaMalloc(&device_buffer_, out_bytes));
            device_buffer_size_ = out_bytes;
            ASSERT_EQ(cudaSuccess, cudaMemset(device_buffer_, 0, out_bytes));
            info.buffer = device_buffer_;
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &info));
    }

    nvimgcodecProcessingStatus_t DecodeAndGetStatus(bool use_sub_stream, int apply_exif_orientation)
    {
        nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params.apply_exif_orientation = apply_exif_orientation;
        nvimgcodecCodeStream_t* cs = use_sub_stream ? &sub_code_stream_ : &code_stream_;
        nvimgcodecStatus_t st = nvimgcodecDecoderDecode(decoder_, cs, &image_, 1, &params, &future_);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, st);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
        nvimgcodecProcessingStatus_t proc_status;
        size_t status_size;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &proc_status, &status_size));
        return proc_status;
    }

    // Copy device buffer to host_buffer_ so the comparison code can read pixels uniformly.
    void StageOutputToHost()
    {
        if (GetParam().kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY) return;
        host_buffer_.assign(device_buffer_size_, 0);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(host_buffer_.data(), device_buffer_, device_buffer_size_, cudaMemcpyDeviceToHost));
    }

    // Decode the entire image (no ROI) with the same backend used by the test. The
    // returned buffer is in I_RGB host memory regardless of the backend's native kind.
    // Returns (host_bytes, dst_width, dst_height).
    std::vector<uint8_t> FullDecodeToHost(const std::string& abs_path, int apply_exif_orientation,
                                          int32_t* out_w, int32_t* out_h)
    {
        nvimgcodecCodeStream_t cs = nullptr;
        LoadImageFromFilename(instance_, cs, abs_path);
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(cs, &info));
        // Apply EXIF dim swap to the OUTPUT shape if appropriate.
        uint32_t out_height = info.plane_info[0].height;
        uint32_t out_width  = info.plane_info[0].width;
        if (apply_exif_orientation && (info.orientation.rotated / 90) % 2 != 0) {
            std::swap(out_height, out_width);
        }
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        info.num_planes = 1;
        info.plane_info[0].width = out_width;
        info.plane_info[0].height = out_height;
        info.plane_info[0].row_stride = static_cast<size_t>(out_width) * 3;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 0;

        std::vector<uint8_t> host(static_cast<size_t>(out_height) * out_width * 3, 0);
        void* dev = nullptr;
        if (GetParam().kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY) {
            info.buffer = host.data();
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        } else {
            if (cudaMalloc(&dev, host.size()) != cudaSuccess) {
                ADD_FAILURE() << "cudaMalloc failed in FullDecodeToHost";
                return {};
            }
            if (cudaMemset(dev, 0, host.size()) != cudaSuccess) {
                ADD_FAILURE() << "cudaMemset failed in FullDecodeToHost";
                cudaFree(dev);
                return {};
            }
            info.buffer = dev;
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }

        nvimgcodecImage_t img = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &img, &info));
        nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params.apply_exif_orientation = apply_exif_orientation;
        nvimgcodecFuture_t fut = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, &cs, &img, 1, &params, &fut));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(fut));
        nvimgcodecProcessingStatus_t st{};
        size_t st_sz = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(fut, &st, &st_sz));
        EXPECT_EQ(st, NVIMGCODEC_PROCESSING_STATUS_SUCCESS) << "full decode failed; status 0x" << std::hex << st;
        if (dev != nullptr) {
            EXPECT_EQ(cudaSuccess, cudaMemcpy(host.data(), dev, host.size(), cudaMemcpyDeviceToHost));
            cudaFree(dev);
        }
        nvimgcodecImageDestroy(img);
        nvimgcodecFutureDestroy(fut);
        nvimgcodecCodeStreamDestroy(cs);
        *out_w = static_cast<int32_t>(out_width);
        *out_h = static_cast<int32_t>(out_height);
        return host;
    }

    nvimgcodecBackend_t backend_{};
    nvimgcodecInstance_t instance_{};
    nvimgcodecDecoder_t decoder_{};
    nvimgcodecCodeStream_t code_stream_{nullptr};
    nvimgcodecCodeStream_t sub_code_stream_{nullptr};
    nvimgcodecImage_t image_{nullptr};
    nvimgcodecFuture_t future_{nullptr};
    std::vector<nvimgcodecExtension_t> extensions_;
    std::vector<uint8_t> host_buffer_;
    void* device_buffer_{nullptr};
    size_t device_buffer_size_{0};
};

// Core regression: a region whose y-extent (500) exceeds the raw codestream height (426)
// must be accepted when apply_exif_orientation=1 because in display coords the image is
// 640 pixels tall.
TEST_P(RoiOrientationIntegrationTest, DisplayCoordRoiThatIsOobInRawCoordsDecodes)
{
    if (!GetParam().can_apply_orientation) {
        GTEST_SKIP() << "Backend " << GetParam().name
                     << " does not apply EXIF orientation; the display-coord ROI contract "
                        "does not apply.";
    }
    const std::string abs_path =
        resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg";

    // Confirm our understanding of the codestream dims (raw, pre-EXIF).
    nvimgcodecCodeStream_t cs_probe = nullptr;
    LoadImageFromFilename(instance_, cs_probe, abs_path);
    nvimgcodecImageInfo_t cs_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(cs_probe, &cs_info));
    ASSERT_EQ(cs_info.plane_info[0].width,  640u);  // raw W
    ASSERT_EQ(cs_info.plane_info[0].height, 426u);  // raw H
    // EXIF orientation 6 (ROTATE_90_CW) maps to nvimgcodec rotated=270 (CCW degrees).
    ASSERT_EQ(cs_info.orientation.rotated, 270);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(cs_probe));

    // Display dims are (W=426, H=640). Pick a region whose y_end exceeds raw_H=426.
    nvimgcodecRegion_t region{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region.start[0] = 0;   region.start[1] = 0;
    region.end[0]   = 500; region.end[1]   = 200;

    PrepareImage(abs_path, /*out_height=*/500, /*out_width=*/200, &region);
    nvimgcodecProcessingStatus_t status = DecodeAndGetStatus(/*use_sub_stream=*/true, /*apply_exif_orientation=*/1);

    ASSERT_EQ(status, NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        << "expected SUCCESS, got 0x" << std::hex << status
        << " — the OOB-in-raw / in-display-bounds ROI was rejected by backend "
        << GetParam().name;

    StageOutputToHost();

    // Same-backend reference: decode the entire image (oriented) and crop in memory.
    // This isolates the test from cross-decoder pixel variance (CPU vs nvjpeg's RGB).
    int32_t ref_w = 0, ref_h = 0;
    auto ref = FullDecodeToHost(abs_path, /*apply_exif_orientation=*/1, &ref_w, &ref_h);
    ASSERT_EQ(ref_h, 640);
    ASSERT_EQ(ref_w, 426);
    const int row_stride = ref_w * 3;
    const int crop_w = region.end[1] - region.start[1];
    const int crop_h = region.end[0] - region.start[0];
    std::vector<uint8_t> ref_crop(static_cast<size_t>(crop_h) * crop_w * 3);
    for (int y = 0; y < crop_h; ++y) {
        std::memcpy(ref_crop.data() + static_cast<size_t>(y) * crop_w * 3,
                    ref.data() + (region.start[0] + y) * row_stride + region.start[1] * 3,
                    static_cast<size_t>(crop_w) * 3);
    }
    ASSERT_EQ(ref_crop.size(), host_buffer_.size());

    // With fancy upsampling disabled (see decoder options in SetUp), nvjpeg's ROI mode
    // matches full-decode + crop exactly on every backend.
    int max_diff = 0;
    for (size_t i = 0; i < host_buffer_.size(); ++i) {
        int d = std::abs(static_cast<int>(host_buffer_[i]) - static_cast<int>(ref_crop[i]));
        max_diff = std::max(max_diff, d);
    }
    EXPECT_EQ(0, max_diff)
        << "ROI decode and full+crop disagree for backend " << GetParam().name
        << " (max pixel diff " << max_diff << ") — the wrong region was likely decoded";
}

// Companion test for the apply_exif_orientation=0 half of the contract: a region in
// raw codestream coordinates that fits the raw dims must decode correctly. The output
// pixels must match a cv::imread reference *without* EXIF rotation, cropped to the same
// raw-coord rectangle. This guards against regressions where the raw-coord region were
// (mis-)interpreted as display coords.
TEST_P(RoiOrientationIntegrationTest, InBoundsRawCoordRoiDecodesWithoutOrientation)
{
    const std::string abs_path =
        resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg";

    // Raw codestream is 640 wide x 426 tall. Pick a region well inside raw bounds.
    nvimgcodecRegion_t region{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region.start[0] = 50;  region.start[1] = 100;
    region.end[0]   = 250; region.end[1]   = 400;

    PrepareImage(abs_path, /*out_height=*/200, /*out_width=*/300, &region);
    nvimgcodecProcessingStatus_t status = DecodeAndGetStatus(/*use_sub_stream=*/true, /*apply_exif_orientation=*/0);
    ASSERT_EQ(status, NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        << "in-bounds raw-coord ROI rejected (status 0x" << std::hex << status
        << ") by backend " << GetParam().name;

    StageOutputToHost();

    // Same-backend reference: full raw decode (apply_exif_orientation=0) then crop.
    int32_t ref_w = 0, ref_h = 0;
    auto ref = FullDecodeToHost(abs_path, /*apply_exif_orientation=*/0, &ref_w, &ref_h);
    ASSERT_EQ(ref_h, 426);
    ASSERT_EQ(ref_w, 640);
    const int row_stride = ref_w * 3;
    const int crop_w = region.end[1] - region.start[1];
    const int crop_h = region.end[0] - region.start[0];
    std::vector<uint8_t> ref_crop(static_cast<size_t>(crop_h) * crop_w * 3);
    for (int y = 0; y < crop_h; ++y) {
        std::memcpy(ref_crop.data() + static_cast<size_t>(y) * crop_w * 3,
                    ref.data() + (region.start[0] + y) * row_stride + region.start[1] * 3,
                    static_cast<size_t>(crop_w) * 3);
    }
    ASSERT_EQ(ref_crop.size(), host_buffer_.size());

    int max_diff = 0;
    for (size_t i = 0; i < host_buffer_.size(); ++i) {
        int d = std::abs(static_cast<int>(host_buffer_[i]) - static_cast<int>(ref_crop[i]));
        max_diff = std::max(max_diff, d);
    }
    EXPECT_EQ(0, max_diff)
        << "raw-coord ROI decode and full+crop disagree for backend " << GetParam().name
        << " (max pixel diff " << max_diff << ") — the wrong raw region was likely decoded";
}

INSTANTIATE_TEST_SUITE_P(AllBackends, RoiOrientationIntegrationTest,
                         ::testing::ValuesIn(kBackendCases),
                         [](const auto& info) { return info.param.name; });

// OOB-after-EXIF: ROI extends past the display image on every edge and must be
// filled with zeros. This guards specifically against a fill that used raw
// codestream dims instead of display dims — for the rotated 640x426 image the
// real-pixel rows [426, 640) would otherwise be classified as fill territory and
// the inner region would not match the full display decode.
//
// Fill is only implemented for device buffers (see fill_out_of_bounds_region),
// so this fixture targets the nvjpeg backends only.
class RoiOobAfterExifTest : public RoiOrientationIntegrationTest {};

TEST_P(RoiOobAfterExifTest, SymmetricFrameOnRotatedImageIsFilledInDisplayCoords)
{
    const std::string abs_path =
        resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg";

    // Display dims (post-EXIF): 426 wide x 640 tall. Extend the ROI 50 px past on
    // every edge so the fill code path runs for top/bottom and left/right.
    constexpr int pad = 50;
    constexpr int disp_h = 640;
    constexpr int disp_w = 426;
    constexpr int out_h = disp_h + 2 * pad;
    constexpr int out_w = disp_w + 2 * pad;

    nvimgcodecRegion_t region{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region.start[0] = -pad;        region.start[1] = -pad;
    region.end[0]   = disp_h + pad; region.end[1]  = disp_w + pad;

    PrepareImage(abs_path, /*out_height=*/out_h, /*out_width=*/out_w, &region);
    nvimgcodecProcessingStatus_t status = DecodeAndGetStatus(/*use_sub_stream=*/true, /*apply_exif_orientation=*/1);
    ASSERT_EQ(status, NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        << "expected SUCCESS, got 0x" << std::hex << status
        << " — symmetric-frame OOB ROI rejected by backend " << GetParam().name;
    StageOutputToHost();

    int32_t ref_w = 0, ref_h = 0;
    auto ref = FullDecodeToHost(abs_path, /*apply_exif_orientation=*/1, &ref_w, &ref_h);
    ASSERT_EQ(ref_h, disp_h);
    ASSERT_EQ(ref_w, disp_w);

    // Inner region must match the full display decode pixel-for-pixel. If the
    // fill ran in raw coords, real-pixel rows in [426, 640) would be zero here.
    int inner_max_diff = 0;
    for (int y = 0; y < disp_h; ++y) {
        const uint8_t* out_row = host_buffer_.data() + static_cast<size_t>(y + pad) * out_w * 3 + pad * 3;
        const uint8_t* ref_row = ref.data() + static_cast<size_t>(y) * disp_w * 3;
        for (int i = 0; i < disp_w * 3; ++i) {
            inner_max_diff = std::max(inner_max_diff, std::abs(static_cast<int>(out_row[i]) - static_cast<int>(ref_row[i])));
        }
    }
    EXPECT_EQ(0, inner_max_diff)
        << "inner region differs from full display decode on backend " << GetParam().name
        << " (max diff " << inner_max_diff << ") — OOB fill likely ran in raw codestream coords";

    // The 50-pixel frame on every side must be zero.
    auto is_zero_row = [&](int y) {
        const uint8_t* row = host_buffer_.data() + static_cast<size_t>(y) * out_w * 3;
        for (int i = 0; i < out_w * 3; ++i) if (row[i] != 0) return false;
        return true;
    };
    auto is_zero_col_span = [&](int y0, int y1, int x0, int x1) {
        for (int y = y0; y < y1; ++y) {
            const uint8_t* row = host_buffer_.data() + static_cast<size_t>(y) * out_w * 3;
            for (int x = x0; x < x1; ++x) {
                for (int c = 0; c < 3; ++c) if (row[x * 3 + c] != 0) return false;
            }
        }
        return true;
    };
    for (int y = 0; y < pad; ++y) EXPECT_TRUE(is_zero_row(y)) << "top fill row " << y << " not zero on " << GetParam().name;
    for (int y = pad + disp_h; y < out_h; ++y) EXPECT_TRUE(is_zero_row(y)) << "bottom fill row " << y << " not zero on " << GetParam().name;
    EXPECT_TRUE(is_zero_col_span(pad, pad + disp_h, 0, pad)) << "left fill not zero on " << GetParam().name;
    EXPECT_TRUE(is_zero_col_span(pad, pad + disp_h, pad + disp_w, out_w)) << "right fill not zero on " << GetParam().name;
}

TEST_P(RoiOobAfterExifTest, RoiExtendsPastDisplayHButFitsRawH)
{
    // The critical regression: ROI y-extent (700) > display_H (640) but the value
    // (700) is also > raw_H (426). A fill that used raw dims would mark rows
    // [426, 700) as OOB and overwrite the real-pixel rows [426, 640) inside the
    // display image. The correct (display-coord) fill marks only [640, 700) OOB.
    const std::string abs_path =
        resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg";

    constexpr int disp_h = 640;
    constexpr int disp_w = 426;
    constexpr int pad_bottom = 60;
    constexpr int out_h = disp_h + pad_bottom;
    constexpr int out_w = disp_w;

    nvimgcodecRegion_t region{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region.start[0] = 0;     region.start[1] = 0;
    region.end[0]   = out_h; region.end[1]   = out_w;

    PrepareImage(abs_path, /*out_height=*/out_h, /*out_width=*/out_w, &region);
    nvimgcodecProcessingStatus_t status = DecodeAndGetStatus(/*use_sub_stream=*/true, /*apply_exif_orientation=*/1);
    ASSERT_EQ(status, NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        << "expected SUCCESS, got 0x" << std::hex << status
        << " — OOB-past-display-H ROI rejected by backend " << GetParam().name;
    StageOutputToHost();

    int32_t ref_w = 0, ref_h = 0;
    auto ref = FullDecodeToHost(abs_path, /*apply_exif_orientation=*/1, &ref_w, &ref_h);
    ASSERT_EQ(ref_h, disp_h);
    ASSERT_EQ(ref_w, disp_w);

    // Rows [0, disp_h) must equal full display decode. Fails if fill used raw_H.
    int inner_max_diff = 0;
    for (int y = 0; y < disp_h; ++y) {
        const uint8_t* out_row = host_buffer_.data() + static_cast<size_t>(y) * out_w * 3;
        const uint8_t* ref_row = ref.data() + static_cast<size_t>(y) * disp_w * 3;
        for (int i = 0; i < disp_w * 3; ++i) {
            inner_max_diff = std::max(inner_max_diff, std::abs(static_cast<int>(out_row[i]) - static_cast<int>(ref_row[i])));
        }
    }
    EXPECT_EQ(0, inner_max_diff)
        << "rows [0, " << disp_h << ") differ from full display decode on backend "
        << GetParam().name << " (max diff " << inner_max_diff
        << ") — OOB fill likely ran in raw codestream coords";

    // Rows [disp_h, out_h) must be zero (filled).
    for (int y = disp_h; y < out_h; ++y) {
        const uint8_t* row = host_buffer_.data() + static_cast<size_t>(y) * out_w * 3;
        for (int i = 0; i < out_w * 3; ++i) {
            EXPECT_EQ(0, row[i]) << "bottom OOB row " << y << " idx " << i << " not zero on " << GetParam().name;
        }
    }
}

// Only the nvjpeg backends implement OOB fill. opencv/libjpeg_turbo reject OOB
// ROIs in canDecode and we skip them.
constexpr BackendCase kOobBackendCases[] = {
    kBackendCases[2], // nvjpeg_cuda
    kBackendCases[3], // nvjpeg_hw
};
INSTANTIATE_TEST_SUITE_P(NvjpegBackends, RoiOobAfterExifTest,
                         ::testing::ValuesIn(kOobBackendCases),
                         [](const auto& info) { return info.param.name; });

// libjpeg-turbo cannot apply EXIF orientation. With apply_exif_orientation=1 on an
// EXIF-rotated JPEG it must refuse via PROCESSING_STATUS_ORIENTATION_UNSUPPORTED. This
// guards against my bounds-check refactor accidentally turning the libjpeg_turbo
// rejection into something else (e.g. by ordering the new effective bounds check after
// the orientation check, or by accidentally reordering status flags).
class LibJpegTurboOrientationRejectTest : public RoiOrientationIntegrationTest {};

TEST_P(LibJpegTurboOrientationRejectTest, RotatedImageWithOrientationOnIsRejected)
{
    const std::string abs_path =
        resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg";

    // An in-bounds raw-coord ROI — the rejection must come from the orientation check,
    // not from any ROI-bounds path.
    nvimgcodecRegion_t region{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region.start[0] = 0;   region.start[1] = 0;
    region.end[0]   = 100; region.end[1]   = 100;

    PrepareImage(abs_path, /*out_height=*/100, /*out_width=*/100, &region);
    nvimgcodecProcessingStatus_t status = DecodeAndGetStatus(/*use_sub_stream=*/true, /*apply_exif_orientation=*/1);
    EXPECT_NE(status, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
    EXPECT_TRUE(status & NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED)
        << "expected ORIENTATION_UNSUPPORTED bit set; got status 0x" << std::hex << status;
}

// Only run the libjpeg_turbo rejection check on the libjpeg_turbo-only backend case.
INSTANTIATE_TEST_SUITE_P(LibJpegTurboOnly, LibJpegTurboOrientationRejectTest,
                         ::testing::Values(kBackendCases[0]),
                         [](const auto& info) { return info.param.name; });

}}  // namespace nvimgcodec::test
