# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pytest as t
from nvidia import nvimgcodec
from utils import img_dir_path

backends_cpu_only = [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]
backends_gpu_only = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)]
default_image_shape = (480, 640, 3)


def reference_image():
    """A real natural image (decoded from an existing JPEG 4:4:4 file), used as a
    smooth source for lossy round-trip tests. Unlike random noise, chroma
    subsampling barely affects it, so round-trip diffs stay small and the
    thresholds below stay tight and meaningful."""
    path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_444.jpg")
    return np.asarray(nvimgcodec.Decoder().decode(path).cpu())

def encode_decode(extension, backends, dtype, shape, max_mean_diff=None, encode_params=None, decode_params=None, image=None):
    encoder = nvimgcodec.Encoder(backends=backends)
    decoder = nvimgcodec.Decoder(backends=backends)

    if image is None:
        image = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype)
    assert image.dtype == dtype
    encoded = encoder.encode(image, extension, params = encode_params)
    assert encoded is not None

    decoded_gpu = decoder.decode(encoded, params=decode_params)
    assert decoded_gpu is not None

    decoded = np.asarray(decoded_gpu.cpu())
    assert decoded.dtype == dtype

    mean_diff = np.abs(image.astype(np.int32) - decoded.astype(np.int32)).mean()

    if max_mean_diff is None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0 and mean_diff / (np.iinfo(dtype).max - np.iinfo(dtype).min) < max_mean_diff

def encode_decode_lossless(extension, backends, dtype, shape):
    assert extension != "jpeg", "currently only lossy jpeg encoding is supported"

    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    encode_decode(extension, backends, dtype, shape, max_mean_diff=None,
                    encode_params=encode_params, decode_params=decode_params)

def encode_decode_lossy(extension, backends, dtype, shape, chroma_subsampling=None):
    assert extension == "jpeg" or extension == "jpeg2k" or extension == "webp"

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    # Use a real (smooth) reference image for the 8-bit codecs so the round-trip
    # diff stays small; the wider-depth jpeg2k variants keep using random data.
    image = reference_image() if dtype == np.uint8 else None

    # JPEG is tested at both 4:4:4 and 4:2:0 (its multi-channel default). 4:2:0
    # subsamples chroma, so it gets a slightly looser bound than 4:4:4.
    encode_params = None
    if extension == "jpeg":
        assert chroma_subsampling is not None, "jpeg requires an explicit chroma_subsampling"
        encode_params = nvimgcodec.EncodeParams(chroma_subsampling=chroma_subsampling)
        max_mean_diff = 0.025 if chroma_subsampling == nvimgcodec.ChromaSubsampling.CSS_420 else 0.02
    elif extension == "webp":
        max_mean_diff = 0.03 # webp uses subsampling
    elif extension == "jpeg2k":
        max_mean_diff = 0.005
    else:
        max_mean_diff = 0.06

    encode_decode(extension, backends, dtype, shape, max_mean_diff=max_mean_diff,
                    encode_params=encode_params, decode_params=decode_params, image=image)

@t.mark.parametrize("extension", ["png", "bmp", "jpeg2k", "pnm", "tiff", "webp"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_uint8_lossless(extension, backends):
    encode_decode_lossless(extension, backends, np.uint8, default_image_shape)

@t.mark.parametrize("extension", ["png", "jpeg2k", "pnm", "tiff"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_uint16_lossless(extension, backends):
    encode_decode_lossless(extension, backends, np.uint16, default_image_shape)

@t.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_only_gpu_lossless(dtype):
    encode_decode_lossless("jpeg2k", backends_gpu_only, dtype, default_image_shape)

@t.mark.parametrize("extension,dtype,chroma_subsampling", [
    ("jpeg", np.uint8, nvimgcodec.ChromaSubsampling.CSS_444),
    ("jpeg", np.uint8, nvimgcodec.ChromaSubsampling.CSS_420),
    ("jpeg2k", np.uint8, None),
    ("jpeg2k", np.uint16, None),
    ("jpeg2k", np.int16, None),
    ("webp", np.uint8, None),
], ids=["jpeg-u8-444", "jpeg-u8-420", "jpeg2k-u8", "jpeg2k-u16", "jpeg2k-i16", "webp-u8"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_lossy(extension, dtype, chroma_subsampling, backends):
    if dtype == np.int16 and backends == backends_cpu_only:
        t.skip("CPU plugins don't support int16")

    encode_decode_lossy(extension, backends, dtype, default_image_shape, chroma_subsampling)

@t.mark.parametrize("extension,dtype,chroma_subsampling", [
    ("jpeg", np.uint8, nvimgcodec.ChromaSubsampling.CSS_444),
    ("jpeg", np.uint8, nvimgcodec.ChromaSubsampling.CSS_420),
    ("jpeg2k", np.uint8, None),
    ("jpeg2k", np.uint16, None),
    ("jpeg2k", np.int16, None),
], ids=["jpeg-u8-444", "jpeg-u8-420", "jpeg2k-u8", "jpeg2k-u16", "jpeg2k-i16"])
def test_only_gpu_lossy(extension, dtype, chroma_subsampling):
    encode_decode_lossy(extension, backends_gpu_only, dtype, default_image_shape, chroma_subsampling)

@t.mark.parametrize("reversible", [True, False])
@t.mark.parametrize("backends", [backends_gpu_only, None]) # TODO: add cpu HT jpeg2k
@t.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_ht_jpeg2k(reversible, backends, dtype):
    encode_params=nvimgcodec.EncodeParams(
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(ht=True),
    )
    if reversible:
        encode_params.quality_type = nvimgcodec.QualityType.LOSSLESS

    max_mean_diff = None if reversible else 0.005

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    encode_decode("jpeg2k", backends, dtype, default_image_shape, max_mean_diff=max_mean_diff,
                    encode_params=encode_params, decode_params=decode_params)

def encode_decode_with_padding(extension, backends, chroma_subsampling=None):
    # Use a real (smooth) image cropped to (200, 100, 3) so the lossy round-trip
    # diff stays small. Dropping some columns turns the row stride into padding.
    backing = np.ascontiguousarray(reference_image()[:200, :100, :3])
    img_rgb = backing[:, 10:80]
    assert img_rgb.shape == (200, 70, 3)
    assert img_rgb.strides == (300, 3, 1)

    encoder = nvimgcodec.Encoder(backends=backends)
    decoder = nvimgcodec.Decoder()

    if extension == "jpeg":
        # JPEG is tested at both 4:4:4 and 4:2:0 (its multi-channel default).
        # 4:2:0 subsamples chroma, so it round-trips with a slightly larger diff
        # than full-resolution 4:4:4.
        assert chroma_subsampling is not None, "jpeg requires an explicit chroma_subsampling"
        params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=95,
                                         chroma_subsampling=chroma_subsampling)
    else:
        params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)

    encoded = encoder.encode(img_rgb, extension, params=params)
    assert encoded is not None

    decoded = np.array(decoder.decode(encoded).cpu())

    mean_diff = np.abs(img_rgb.astype(np.int32) - decoded.astype(np.int32)).mean()
    if extension == "jpeg":
        max_mean_diff = 3 if chroma_subsampling == nvimgcodec.ChromaSubsampling.CSS_420 else 2
        assert mean_diff < max_mean_diff
    else:
        assert mean_diff == 0

@t.mark.parametrize("extension,chroma_subsampling", [
    ("jpeg", nvimgcodec.ChromaSubsampling.CSS_444),
    ("jpeg", nvimgcodec.ChromaSubsampling.CSS_420),
    ("jpeg2k", None),
], ids=["jpeg-444", "jpeg-420", "jpeg2k"])
@t.mark.parametrize("backends", [backends_gpu_only])
def test_encode_decode_with_padding_gpu(extension, backends, chroma_subsampling):
    encode_decode_with_padding(extension, backends, chroma_subsampling)

@t.mark.parametrize("extension,chroma_subsampling", [
    ("jpeg", nvimgcodec.ChromaSubsampling.CSS_444),
    ("jpeg", nvimgcodec.ChromaSubsampling.CSS_420),
    ("png", None),
    ("bmp", None),
    ("jpeg2k", None),
    ("pnm", None),
    ("tiff", None),
    ("webp", None),
], ids=["jpeg-444", "jpeg-420", "png", "bmp", "jpeg2k", "pnm", "tiff", "webp"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_encode_decode_with_padding(extension, backends, chroma_subsampling):
    encode_decode_with_padding(extension, backends, chroma_subsampling)
