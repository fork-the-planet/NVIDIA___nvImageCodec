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

#include "imgproc/region_orientation.h"

namespace nvimgcodec {
namespace {

nvimgcodecOrientation_t make_orient(int rotated, bool flip_x, bool flip_y) {
    nvimgcodecOrientation_t o{};
    o.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION;
    o.struct_size = sizeof(nvimgcodecOrientation_t);
    o.rotated = rotated;
    o.flip_x = flip_x ? 1 : 0;
    o.flip_y = flip_y ? 1 : 0;
    return o;
}

nvimgcodecRegion_t make_region(int y0, int x0, int y1, int x1) {
    nvimgcodecRegion_t r{};
    r.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
    r.struct_size = sizeof(nvimgcodecRegion_t);
    r.ndim = 2;
    r.start[0] = y0;
    r.start[1] = x0;
    r.end[0] = y1;
    r.end[1] = x1;
    return r;
}

void expect_dims(std::pair<uint32_t, uint32_t> got, uint32_t w, uint32_t h) {
    EXPECT_EQ(got.first, w);
    EXPECT_EQ(got.second, h);
}

TEST(RegionOrientation, OrientedDims) {
    expect_dims(oriented_dims(768, 1024, make_orient(0,   false, false)), 768, 1024);
    expect_dims(oriented_dims(768, 1024, make_orient(90,  false, false)), 1024, 768);
    expect_dims(oriented_dims(768, 1024, make_orient(180, false, false)), 768, 1024);
    expect_dims(oriented_dims(768, 1024, make_orient(270, false, false)), 1024, 768);
    // Flips do not change the dims.
    expect_dims(oriented_dims(768, 1024, make_orient(0,   true,  true)),  768, 1024);
}

TEST(RegionOrientation, IsOutOfBoundsEffective_RespectsApply) {
    auto o = make_orient(90, false, false);
    // raw 768x1024 -> display 1024x768. A region 0..1024 in x is in bounds for display
    // (W_d=1024) but out of bounds for raw (W_r=768).
    auto r = make_region(0, 0, 768, 1024);
    EXPECT_FALSE(is_region_out_of_bounds_effective(r, o, 768, 1024, /*apply=*/true));
    EXPECT_TRUE(is_region_out_of_bounds_effective(r, o, 768, 1024, /*apply=*/false));
}

}  // namespace
}  // namespace nvimgcodec
