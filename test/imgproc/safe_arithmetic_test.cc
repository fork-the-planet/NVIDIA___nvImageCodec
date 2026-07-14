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
#include <cstdint>
#include <limits>
#include "imgproc/safe_arithmetic.h"

namespace nvimgcodec {

constexpr size_t kMax = std::numeric_limits<size_t>::max();

// ---------------------------------------------------------------------------
// SafeMulSizeT
// ---------------------------------------------------------------------------

TEST(SafeArithmetic, MulBasic) {
    size_t out = 123;
    EXPECT_TRUE(SafeMulSizeT(6, 7, out));
    EXPECT_EQ(out, 42u);
}

TEST(SafeArithmetic, MulByZero) {
    size_t out = 123;
    EXPECT_TRUE(SafeMulSizeT(0, kMax, out));
    EXPECT_EQ(out, 0u);
    EXPECT_TRUE(SafeMulSizeT(kMax, 0, out));
    EXPECT_EQ(out, 0u);
}

TEST(SafeArithmetic, MulByOne) {
    size_t out = 0;
    EXPECT_TRUE(SafeMulSizeT(kMax, 1, out));
    EXPECT_EQ(out, kMax);
}

TEST(SafeArithmetic, MulAtBoundary) {
    // kMax is odd, so kMax/2 * 2 == kMax - 1 (no overflow).
    size_t out = 0;
    EXPECT_TRUE(SafeMulSizeT(kMax / 2, 2, out));
    EXPECT_EQ(out, kMax - 1);
}

TEST(SafeArithmetic, MulOverflow) {
    size_t out = 0;
    EXPECT_FALSE(SafeMulSizeT(kMax, 2, out));
    EXPECT_FALSE(SafeMulSizeT(kMax / 2 + 1, 2, out));
    EXPECT_FALSE(SafeMulSizeT(size_t{1} << 33, size_t{1} << 33, out));
}

// ---------------------------------------------------------------------------
// SafeMul3SizeT
// ---------------------------------------------------------------------------

TEST(SafeArithmetic, Mul3Basic) {
    size_t out = 0;
    EXPECT_TRUE(SafeMul3SizeT(2, 3, 4, out));
    EXPECT_EQ(out, 24u);
}

TEST(SafeArithmetic, Mul3OverflowFirstStep) {
    size_t out = 0;
    // a*b overflows before c is applied.
    EXPECT_FALSE(SafeMul3SizeT(kMax, 2, 1, out));
}

TEST(SafeArithmetic, Mul3OverflowSecondStep) {
    size_t out = 0;
    // a*b fits, but (a*b)*c overflows.
    EXPECT_FALSE(SafeMul3SizeT(kMax / 2, 2, 2, out));
}

// ---------------------------------------------------------------------------
// SafeAddSizeT
// ---------------------------------------------------------------------------

TEST(SafeArithmetic, AddBasic) {
    size_t out = 0;
    EXPECT_TRUE(SafeAddSizeT(40, 2, out));
    EXPECT_EQ(out, 42u);
}

TEST(SafeArithmetic, AddAtBoundary) {
    size_t out = 0;
    EXPECT_TRUE(SafeAddSizeT(kMax - 1, 1, out));
    EXPECT_EQ(out, kMax);
}

TEST(SafeArithmetic, AddOverflow) {
    size_t out = 0;
    EXPECT_FALSE(SafeAddSizeT(kMax, 1, out));
    EXPECT_FALSE(SafeAddSizeT(kMax, kMax, out));
}

// ---------------------------------------------------------------------------
// SafePlaneByteSize
// ---------------------------------------------------------------------------

TEST(SafeArithmetic, PlaneByteSizeBasic) {
    nvimgcodecImagePlaneInfo_t plane{};
    plane.row_stride = 640;
    plane.height = 480;
    size_t out = 0;
    EXPECT_TRUE(SafePlaneByteSize(plane, out));
    EXPECT_EQ(out, 640u * 480u);
}

TEST(SafeArithmetic, PlaneByteSizeOverflow) {
    nvimgcodecImagePlaneInfo_t plane{};
    plane.row_stride = kMax / 2;
    plane.height = 4;
    size_t out = 0;
    EXPECT_FALSE(SafePlaneByteSize(plane, out));
}

// ---------------------------------------------------------------------------
// SafePlaneByteOffsets
// ---------------------------------------------------------------------------

namespace {
nvimgcodecImageInfo_t MakeInfo(uint32_t num_planes, const size_t* strides, const uint32_t* heights) {
    nvimgcodecImageInfo_t info{};
    info.num_planes = num_planes;
    for (uint32_t p = 0; p < num_planes; ++p) {
        info.plane_info[p].row_stride = strides[p];
        info.plane_info[p].height = heights[p];
    }
    return info;
}
}  // namespace

TEST(SafeArithmetic, PlaneOffsetsUniform) {
    const size_t strides[3] = {100, 100, 100};
    const uint32_t heights[3] = {10, 10, 10};
    auto info = MakeInfo(3, strides, heights);
    size_t off[NVIMGCODEC_MAX_NUM_PLANES] = {};
    ASSERT_TRUE(SafePlaneByteOffsets(info, off));
    EXPECT_EQ(off[0], 0u);
    EXPECT_EQ(off[1], 1000u);
    EXPECT_EQ(off[2], 2000u);
}

TEST(SafeArithmetic, PlaneOffsetsNonUniform) {
    // e.g. luma plane larger than chroma planes (subsampled layout / padding).
    const size_t strides[3] = {200, 100, 100};
    const uint32_t heights[3] = {20, 10, 10};
    auto info = MakeInfo(3, strides, heights);
    size_t off[NVIMGCODEC_MAX_NUM_PLANES] = {};
    ASSERT_TRUE(SafePlaneByteOffsets(info, off));
    EXPECT_EQ(off[0], 0u);
    EXPECT_EQ(off[1], 200u * 20u);          // 4000
    EXPECT_EQ(off[2], 200u * 20u + 100u * 10u);  // 5000
}

TEST(SafeArithmetic, PlaneOffsetsSinglePlane) {
    const size_t strides[1] = {123};
    const uint32_t heights[1] = {7};
    auto info = MakeInfo(1, strides, heights);
    size_t off[NVIMGCODEC_MAX_NUM_PLANES] = {};
    ASSERT_TRUE(SafePlaneByteOffsets(info, off));
    EXPECT_EQ(off[0], 0u);
}

TEST(SafeArithmetic, PlaneOffsetsProductOverflow) {
    // A single plane whose row_stride * height overflows size_t.
    const size_t strides[1] = {kMax / 2};
    const uint32_t heights[1] = {4};
    auto info = MakeInfo(1, strides, heights);
    size_t off[NVIMGCODEC_MAX_NUM_PLANES] = {};
    EXPECT_FALSE(SafePlaneByteOffsets(info, off));
}

TEST(SafeArithmetic, PlaneOffsetsRunningSumOverflow) {
    // Each plane fits, but the cumulative offset overflows.
    const size_t strides[2] = {kMax / 3 * 2, kMax / 3 * 2};
    const uint32_t heights[2] = {1, 1};
    auto info = MakeInfo(2, strides, heights);
    size_t off[NVIMGCODEC_MAX_NUM_PLANES] = {};
    EXPECT_FALSE(SafePlaneByteOffsets(info, off));
}

}  // namespace nvimgcodec
