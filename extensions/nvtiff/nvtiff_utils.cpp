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

#include "nvtiff_utils.h"
#include "error_handling.h"
#include "tiff_utils.h"
#include <nvtiff.h>
#include <stdexcept>

namespace nvtiff {

NvtiffVersion get_nvtiff_version() {
    NvtiffVersion v;
    if (NVTIFF_STATUS_SUCCESS == nvtiffGetProperty(MAJOR_VERSION, &v.major_ver) &&
        NVTIFF_STATUS_SUCCESS == nvtiffGetProperty(MINOR_VERSION, &v.minor_ver) &&
        NVTIFF_STATUS_SUCCESS == nvtiffGetProperty(PATCH_LEVEL, &v.patch_ver)) {
        v.valid = true;
    }
    return v;
}

size_t resolve_ifd_selection(nvtiffStream_t nvtiff_stream, const nvimgcodecCodeStreamInfo_t& codestream_info)
{
    XM_CHECK_NULL(nvtiff_stream);

    nvtiffStreamHeader_t header{};
    XM_CHECK_NVTIFF(nvtiffStreamGetHeader(nvtiff_stream, &header));
    if (header.first_ifd_offset == NVTIFF_NO_IMAGE) {
        throw std::runtime_error("TIFF stream has no images");
    }

    const auto* view = codestream_info.code_stream_view;
    const size_t target_index = view ? view->image_idx : 0;
    const size_t bitstream_offset = view ? view->bitstream_offset : 0;
    if (bitstream_offset != 0 && target_index != 0) {
        throw std::runtime_error("image_idx cannot be combined with bitstream_offset for TIFF IFD views");
    }

    // The nvImageCodec TIFF parser caches image_idx -> IFD offset. Prefer it
    // to avoid re-walking the root IFD chain for every image-indexed sample.
    if (const auto* tiff_ext = nvimgcodec::findInfoTiffExt(&codestream_info); tiff_ext && tiff_ext->ifd_offset != 0) {
        return tiff_ext->ifd_offset;
    }

    size_t ifd_offset = bitstream_offset != 0 ? bitstream_offset : header.first_ifd_offset;
    for (size_t i = 0; i < target_index; ++i) {
        size_t next_ifd_offset = NVTIFF_NO_IMAGE;
        XM_CHECK_NVTIFF(nvtiffStreamGetNextIFDOffset(nvtiff_stream, ifd_offset, &next_ifd_offset));
        if (next_ifd_offset == NVTIFF_NO_IMAGE) {
            throw std::runtime_error("Requested image index is outside the selected TIFF IFD range");
        }
        ifd_offset = next_ifd_offset;
    }

    return ifd_offset;
}

size_t resolve_ifd_selection(nvtiffStream_t nvtiff_stream, const nvimgcodecCodeStreamDesc_t* code_stream)
{
    XM_CHECK_NULL(code_stream);
    nvimgcodecCodeStreamInfoTiffExt_t tiff_ext{
        NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr};
    nvimgcodecCodeStreamInfo_t codestream_info{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), &tiff_ext};
    if (code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info) != NVIMGCODEC_STATUS_SUCCESS) {
        throw std::runtime_error("Could not get code stream info.");
    }

    return resolve_ifd_selection(nvtiff_stream, codestream_info);
}

}
