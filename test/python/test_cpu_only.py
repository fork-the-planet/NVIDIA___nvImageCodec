# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""End-to-end CPU-only smoke suite.

Each test wires the codec library with ``device_id=NVIMGCODEC_DEVICE_CPU_ONLY``
and ``backends=[Backend(BackendKind.CPU_ONLY)]`` and exercises a slice of the
public API that should remain functional on a host without a CUDA driver.
The driverless guarantee is verified by where this file runs (the no-GPU CI
lane); locally on a GPU host the same tests still pass.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nvidia import nvimgcodec

CPU_ONLY = nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY
CPU_ONLY_BACKENDS = [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]

img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../resources"))


def _img(rel: str) -> str:
    """Path to a sample resource; fail if missing."""
    p = os.path.join(img_dir_path, rel)
    assert os.path.exists(p), f"resource missing: {p}"
    return p


def _cpu_decoder() -> nvimgcodec.Decoder:
    return nvimgcodec.Decoder(device_id=CPU_ONLY, backends=CPU_ONLY_BACKENDS)


def _cpu_encoder() -> nvimgcodec.Encoder:
    return nvimgcodec.Encoder(device_id=CPU_ONLY, backends=CPU_ONLY_BACKENDS)


# ---------------------------------------------------------------------------
# Module-level constant export
# ---------------------------------------------------------------------------

def test_device_constants_exported():
    assert nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY == -99999
    assert nvimgcodec.NVIMGCODEC_DEVICE_CURRENT == -1


def test_decoder_constructs_with_cpu_only_symbol():
    """The Decoder must accept the exported sentinel directly (no magic int).

    Replaces an inline ``python -c`` smoke step from the no-GPU CI lane: any
    failure to load the binding, export the symbol, or construct a CPU-only
    decoder fails the test instead of a free-floating shell snippet.
    """
    decoder = _cpu_decoder()
    assert decoder.device_id == CPU_ONLY


# ---------------------------------------------------------------------------
# CodeStream parsing — pure CPU, no decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel,expected_codec", [
    ("jpeg/padlock-406986_640_444.jpg", "jpeg"),
    ("png/cat-1245673_640.png", "png"),
    ("tiff/cat-300572_640.tiff", "tiff"),
    ("bmp/cat-111793_640.bmp", "bmp"),
    ("jpeg2k/cat-111793_640.jp2", "jpeg2k"),
])
def test_code_stream_from_path(rel, expected_codec):
    cs = nvimgcodec.CodeStream(_img(rel))
    assert cs.codec_name == expected_codec
    assert cs.width > 0 and cs.height > 0


def test_code_stream_from_bytes():
    with open(_img("jpeg/padlock-406986_640_444.jpg"), "rb") as f:
        data = f.read()
    cs = nvimgcodec.CodeStream(data)
    assert cs.codec_name == "jpeg"
    assert cs.width > 0 and cs.height > 0


# ---------------------------------------------------------------------------
# Decode — single + batch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel,channels", [
    ("jpeg/padlock-406986_640_444.jpg", 3),
    ("png/cat-1245673_640.png", 3),
    ("tiff/cat-300572_640.tiff", 3),
    ("bmp/cat-111793_640.bmp", 3),
    ("jpeg2k/cat-111793_640.jp2", 3),
])
def test_decode_read(rel, channels):
    img = _cpu_decoder().read(_img(rel))
    assert img is not None
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert img.shape[2] == channels


def test_decode_batch_read():
    paths = [
        _img("jpeg/padlock-406986_640_444.jpg"),
        _img("png/cat-1245673_640.png"),
        _img("tiff/cat-300572_640.tiff"),
    ]
    imgs = _cpu_decoder().read(paths)
    assert len(imgs) == len(paths)
    for img in imgs:
        assert img is not None
        assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST


def test_decode_from_bytes():
    with open(_img("jpeg/padlock-406986_640_444.jpg"), "rb") as f:
        data = f.read()
    img = _cpu_decoder().decode(data)
    assert img is not None
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST


# ---------------------------------------------------------------------------
# Decode parameter surface — DecodeParams construction is CPU-only by nature
# ---------------------------------------------------------------------------

def test_decode_params_construction():
    p = nvimgcodec.DecodeParams(allow_any_depth=True, color_spec=nvimgcodec.ColorSpec.SRGB)
    assert p.allow_any_depth is True
    assert p.color_spec == nvimgcodec.ColorSpec.SRGB


def test_decode_with_grayscale_color_spec():
    img = _cpu_decoder().read(
        _img("jpeg/padlock-406986_640_444.jpg"),
        params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY),
    )
    assert img is not None
    # Grayscale output is single-channel.
    assert img.shape[2] == 1


# ---------------------------------------------------------------------------
# Image construction from a numpy array (CPU buffer protocol)
# ---------------------------------------------------------------------------

def test_image_from_numpy_array():
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    img = nvimgcodec.as_image(arr)
    assert img is not None
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert tuple(img.shape) == (480, 640, 3)


# ---------------------------------------------------------------------------
# Encode — single image, varying formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", ["jpeg", "png", "bmp", "pnm", "tiff", "jpeg2k", "webp"])
def test_encode_decode_roundtrip(ext):
    src = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    encoded = _cpu_encoder().encode(src, ext)
    assert encoded is not None
    assert encoded.pin_memory is False

    decoded = _cpu_decoder().decode(encoded)
    assert decoded is not None
    assert decoded.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert tuple(decoded.shape)[:2] == src.shape[:2]


def test_encode_lossless_uint16_png():
    src = np.random.randint(0, 65536, (240, 320, 3), dtype=np.uint16)
    params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded = _cpu_encoder().encode(src, "png", params=params)
    assert encoded is not None

    decoded = _cpu_decoder().decode(
        encoded, params=nvimgcodec.DecodeParams(allow_any_depth=True))
    assert decoded is not None
    np.testing.assert_array_equal(np.asarray(decoded), src)


# ---------------------------------------------------------------------------
# Negative paths — corrupted input must surface as an error or None, not crash
# ---------------------------------------------------------------------------

def test_decode_corrupted_jpeg_returns_none_or_raises():
    bogus = b"\xff\xd8\xff" + b"\x00" * 256  # broken JPEG header
    decoder = _cpu_decoder()
    try:
        out = decoder.decode(bogus)
    except Exception:
        return
    assert out is None
