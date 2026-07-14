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

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

struct NpyArray
{
    std::string descr;
    std::vector<size_t> shape;
    std::vector<uint16_t> samples;
};

inline std::string ParseNpyHeaderString(const std::string& header, const std::string& key)
{
    const auto key_pos = header.find("'" + key + "'");
    if (key_pos == std::string::npos) {
        return {};
    }
    const auto value_begin = header.find('\'', header.find(':', key_pos));
    if (value_begin == std::string::npos) {
        return {};
    }
    const auto value_end = header.find('\'', value_begin + 1);
    if (value_end == std::string::npos) {
        return {};
    }
    return header.substr(value_begin + 1, value_end - value_begin - 1);
}

inline std::vector<size_t> ParseNpyShape(const std::string& header)
{
    std::vector<size_t> shape;
    const auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        return shape;
    }
    const auto shape_begin = header.find('(', shape_pos);
    const auto shape_end = header.find(')', shape_begin);
    if (shape_begin == std::string::npos || shape_end == std::string::npos) {
        return shape;
    }

    std::stringstream shape_stream(header.substr(shape_begin + 1, shape_end - shape_begin - 1));
    while (shape_stream.good()) {
        size_t dim = 0;
        shape_stream >> dim;
        if (!shape_stream.fail()) {
            shape.push_back(dim);
        }
        shape_stream.clear();
        shape_stream.ignore(1, ',');
    }
    return shape;
}

inline void LoadNpyReference(const std::string& file_path, NpyArray* ref)
{
    std::ifstream input(file_path, std::ios::binary);
    ASSERT_TRUE(input.is_open()) << file_path;

    char magic[6] = {};
    input.read(magic, sizeof(magic));
    ASSERT_EQ(0, std::memcmp(magic, "\x93NUMPY", sizeof(magic))) << file_path;

    uint8_t major = 0;
    uint8_t minor = 0;
    input.read(reinterpret_cast<char*>(&major), sizeof(major));
    input.read(reinterpret_cast<char*>(&minor), sizeof(minor));
    ASSERT_EQ(1, major) << file_path;
    ASSERT_EQ(0, minor) << file_path;

    uint8_t header_len_bytes[2] = {};
    input.read(reinterpret_cast<char*>(header_len_bytes), sizeof(header_len_bytes));
    const uint16_t header_len = static_cast<uint16_t>(header_len_bytes[0]) | (static_cast<uint16_t>(header_len_bytes[1]) << 8);

    std::string header(header_len, '\0');
    input.read(header.data(), header.size());

    ref->descr = ParseNpyHeaderString(header, "descr");
    ref->shape = ParseNpyShape(header);
    ASSERT_EQ(3, ref->shape.size()) << file_path;
    ASSERT_EQ(1, ref->shape[2]) << file_path;
    ASSERT_TRUE(header.find("'fortran_order': False") != std::string::npos) << file_path;

    const size_t sample_count = ref->shape[0] * ref->shape[1] * ref->shape[2];
    if (ref->descr == "|u1") {
        std::vector<uint8_t> data(sample_count);
        input.read(reinterpret_cast<char*>(data.data()), data.size());
        ASSERT_EQ(static_cast<std::streamsize>(data.size()), input.gcount()) << file_path;
        ref->samples.resize(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            ref->samples[i] = data[i];
        }
    } else if (ref->descr == "<u2") {
        std::vector<uint8_t> data(sample_count * 2);
        input.read(reinterpret_cast<char*>(data.data()), data.size());
        ASSERT_EQ(static_cast<std::streamsize>(data.size()), input.gcount()) << file_path;
        ref->samples.resize(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            ref->samples[i] = static_cast<uint16_t>(data[2 * i]) | (static_cast<uint16_t>(data[2 * i + 1]) << 8);
        }
    } else {
        FAIL() << "Unsupported NPY dtype " << ref->descr << " in " << file_path;
    }
}

inline void AssertDecodedYPlaneMatchesReference(
    const std::string& reference_file, const nvimgcodecImageInfo_t& image_info, const std::vector<unsigned char>& image_buffer)
{
    NpyArray ref;
    ASSERT_NO_FATAL_FAILURE(LoadNpyReference(reference_file, &ref));
    ASSERT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, image_info.plane_info[0].sample_type);
    ASSERT_EQ(ref.shape[0], image_info.plane_info[0].height);
    ASSERT_EQ(ref.shape[1], image_info.plane_info[0].width);
    ASSERT_EQ(0, image_info.plane_info[0].row_stride % sizeof(uint16_t));
    ASSERT_LE((image_info.plane_info[0].height - 1) * image_info.plane_info[0].row_stride +
            image_info.plane_info[0].width * sizeof(uint16_t),
        image_buffer.size());

    const auto* decoded = reinterpret_cast<const uint16_t*>(image_buffer.data());
    const size_t decoded_row_stride = image_info.plane_info[0].row_stride / sizeof(uint16_t);
    for (size_t y = 0; y < ref.shape[0]; ++y) {
        for (size_t x = 0; x < ref.shape[1]; ++x) {
            ASSERT_EQ(ref.samples[y * ref.shape[1] + x], decoded[y * decoded_row_stride + x])
                << "reference=" << reference_file << " y=" << y << " x=" << x;
        }
    }
}

inline std::string ReferenceNpyPath(const std::string& image_file)
{
    std::string reference_file = resources_dir + image_file;
    const auto extension_pos = reference_file.rfind('.');
    if (extension_pos == std::string::npos) {
        return reference_file + ".npy";
    }
    reference_file.replace(extension_pos, std::string::npos, ".npy");
    return reference_file;
}

}} // namespace nvimgcodec::test
