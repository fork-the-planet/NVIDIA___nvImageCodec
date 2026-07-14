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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <cstring>
#include <string>

#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

namespace {

static const char* kMultiPageTiff = "/tiff/multi_page.tif";

nvimgcodecCodeStreamView_t MakeCodeStreamView(size_t image_idx = 0, size_t bitstream_offset = 0)
{
    nvimgcodecRegion_t region{};
    region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
    region.struct_size = sizeof(nvimgcodecRegion_t);

    nvimgcodecCodeStreamView_t view{};
    view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
    view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
    view.image_idx = image_idx;
    view.region = region;
    view.bitstream_offset = bitstream_offset;
    return view;
}

nvimgcodecCodeStreamInfo_t MakeCodeStreamInfo(void* struct_next = nullptr)
{
    nvimgcodecCodeStreamInfo_t info{};
    info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO;
    info.struct_size = sizeof(nvimgcodecCodeStreamInfo_t);
    info.struct_next = struct_next;
    return info;
}

nvimgcodecCodeStreamInfoTiffExt_t MakeTiffExt()
{
    nvimgcodecCodeStreamInfoTiffExt_t tiff_ext{};
    tiff_ext.struct_type = NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO;
    tiff_ext.struct_size = sizeof(nvimgcodecCodeStreamInfoTiffExt_t);
    return tiff_ext;
}

} // namespace

class BitstreamOffsetTest : public ::testing::Test
{
  public:
    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{};
        create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
        create_info.load_builtin_modules = 1;
        create_info.load_extension_modules = 0;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));
    }

    void TearDown() override { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_)); }

    std::string TiffPath() const { return resources_dir + kMultiPageTiff; }

    nvimgcodecInstance_t instance_ = nullptr;
};

TEST_F(BitstreamOffsetTest, RootTiffNumImagesIsExact)
{
    nvimgcodecCodeStream_t code_stream = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromFile(instance_, &code_stream, TiffPath().c_str(), nullptr));

    auto tiff_ext = MakeTiffExt();
    auto info = MakeCodeStreamInfo(&tiff_ext);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(code_stream, &info));
    EXPECT_GT(info.num_images, 1u);
    EXPECT_GT(tiff_ext.ifd_offset, 0u);
    EXPECT_GT(tiff_ext.next_ifd_offset, 0u);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
}

TEST_F(BitstreamOffsetTest, CreationViewRejectsImageIdx)
{
    nvimgcodecCodeStream_t code_stream = nullptr;
    auto view = MakeCodeStreamView(/*image_idx=*/1);
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateFromFile(instance_, &code_stream, TiffPath().c_str(), &view));
    EXPECT_EQ(nullptr, code_stream);
}

TEST_F(BitstreamOffsetTest, CreationViewRejectsRegion)
{
    nvimgcodecCodeStream_t code_stream = nullptr;
    auto view = MakeCodeStreamView();
    view.region.ndim = 2;
    view.region.start[0] = 0;
    view.region.start[1] = 0;
    view.region.end[0] = 16;
    view.region.end[1] = 16;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateFromFile(instance_, &code_stream, TiffPath().c_str(), &view));
    EXPECT_EQ(nullptr, code_stream);
}

TEST_F(BitstreamOffsetTest, RejectsNestedNonzeroImageIdxBeforeParentParse)
{
    nvimgcodecCodeStream_t root_stream = nullptr;
    nvimgcodecCodeStream_t page_stream = nullptr;
    nvimgcodecCodeStream_t nested_stream = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromFile(instance_, &root_stream, TiffPath().c_str(), nullptr));

    auto page_view = MakeCodeStreamView(/*image_idx=*/1);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetSubCodeStream(root_stream, &page_stream, &page_view));

    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamGetSubCodeStream(page_stream, &nested_stream, &page_view));
    EXPECT_EQ(nullptr, nested_stream);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(page_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(root_stream));
}

TEST_F(BitstreamOffsetTest, RejectsOutOfRangeImageIdxAtSubstreamCreation)
{
    nvimgcodecCodeStream_t root_stream = nullptr;
    nvimgcodecCodeStream_t invalid_stream = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromFile(instance_, &root_stream, TiffPath().c_str(), nullptr));

    auto invalid_view = MakeCodeStreamView(/*image_idx=*/100);
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamGetSubCodeStream(root_stream, &invalid_stream, &invalid_view));
    EXPECT_EQ(nullptr, invalid_stream);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(root_stream));
}

TEST_F(BitstreamOffsetTest, NestedZeroImageIdxKeepsParentSelectionBeforeParentParse)
{
    nvimgcodecCodeStream_t root_stream = nullptr;
    nvimgcodecCodeStream_t page_stream = nullptr;
    nvimgcodecCodeStream_t child_stream = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromFile(instance_, &root_stream, TiffPath().c_str(), nullptr));

    auto page1_view = MakeCodeStreamView(/*image_idx=*/1);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetSubCodeStream(root_stream, &page_stream, &page1_view));

    auto page0_view = MakeCodeStreamView(/*image_idx=*/0);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetSubCodeStream(page_stream, &child_stream, &page0_view));

    auto parent_tiff_ext = MakeTiffExt();
    auto parent_info = MakeCodeStreamInfo(&parent_tiff_ext);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(page_stream, &parent_info));

    auto child_tiff_ext = MakeTiffExt();
    auto child_info = MakeCodeStreamInfo(&child_tiff_ext);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(child_stream, &child_info));

    EXPECT_EQ(child_tiff_ext.ifd_offset, parent_tiff_ext.ifd_offset);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(child_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(page_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(root_stream));
}

TEST_F(BitstreamOffsetTest, SubstreamViewDoesNotExposeCallerOwnedStructNext)
{
    nvimgcodecCodeStream_t root_stream = nullptr;
    nvimgcodecCodeStream_t child_stream = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromFile(instance_, &root_stream, TiffPath().c_str(), nullptr));

    int sentinel = 123;
    auto view = MakeCodeStreamView(/*image_idx=*/0);
    view.struct_next = &sentinel;
    view.region.struct_next = &sentinel;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetSubCodeStream(root_stream, &child_stream, &view));

    auto info = MakeCodeStreamInfo();
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(child_stream, &info));
    ASSERT_NE(nullptr, info.code_stream_view);
    EXPECT_EQ(nullptr, info.code_stream_view->struct_next);
    EXPECT_EQ(nullptr, info.code_stream_view->region.struct_next);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(child_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(root_stream));
}

}} // namespace nvimgcodec::test
