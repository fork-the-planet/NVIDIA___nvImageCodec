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

"""Tests for the enable_roi_fancy_upsampling option in nvjpeg_cuda_decoder and
nvjpeg_hw_decoder.

Behaviour under test:
  - enable_roi_fancy_upsampling=0 causes the decoder to reject (canDecode returns
    ROI_UNSUPPORTED) ROI requests on subsampled (non-4:4:4, non-grayscale) images when
    fancy upsampling is also on.  With no other backend available the decode returns None.
  - 4:4:4 images are not affected by this option (no chroma upsampling occurs).
  - Grayscale images are not affected by this option (no chroma upsampling occurs).
  - Full-image decodes (no ROI) are not affected.
  - enable_roi_fancy_upsampling=1 allows ROI decode on subsampled images.
  - The default equals 1 on nvJPEG >= 13.2 and 0 on older versions.

Backend isolation:
  - CUDA decoder tests use BackendKind.HYBRID_CPU_GPU only.  libjpeg-turbo
    registers as CPU_ONLY so it is excluded; if nvjpeg_cuda rejects an image
    there is no fallback and decode() returns None.
  - HW decoder tests use BackendKind.HW_GPU_ONLY only and are skipped when no
    hardware JPEG engine is present on the machine.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from nvidia import nvimgcodec
from utils import img_dir_path, get_nvjpeg_ver, NVJPEG_WITH_FIXED_UPSAMPLING_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUBSAMPLED_IMG = os.path.join(img_dir_path, "jpeg", "padlock-406986_640_420.jpg")
_444_IMG        = os.path.join(img_dir_path, "jpeg", "padlock-406986_640_444.jpg")
_GRAY_IMG       = os.path.join(img_dir_path, "jpeg", "padlock-406986_640_gray.jpg")

# A small region that comfortably fits inside both test images (640x426 raw).
_ROI = nvimgcodec.Region(100, 100, 300, 400)


def _make_decoder(backend_kind: nvimgcodec.BackendKind, extra_options: str = "") -> nvimgcodec.Decoder:
    """Create a single-backend decoder, skipping the test if the backend is unavailable."""
    try:
        return nvimgcodec.Decoder(
            backends=[nvimgcodec.Backend(backend_kind)],
            options=extra_options,
        )
    except RuntimeError as exc:
        pytest.skip(f"Backend {backend_kind} not available: {exc}")


def _roi_decode(decoder: nvimgcodec.Decoder, image_path: str) -> np.ndarray | None:
    cs = nvimgcodec.CodeStream(image_path)
    sub = cs.get_sub_code_stream(region=_ROI)
    result = decoder.read(sub)
    if result is None:
        return None
    return np.asarray(result.cpu())


def _full_decode(decoder: nvimgcodec.Decoder, image_path: str) -> np.ndarray | None:
    result = decoder.read(image_path)
    if result is None:
        return None
    return np.asarray(result.cpu())


# ---------------------------------------------------------------------------
# Parametrisation: (test-id, BackendKind, decoder-option-prefix)
# ---------------------------------------------------------------------------

_BACKENDS = [
    pytest.param(
        nvimgcodec.BackendKind.HYBRID_CPU_GPU,
        "nvjpeg_cuda_decoder",
        id="cuda",
    ),
    pytest.param(
        nvimgcodec.BackendKind.HW_GPU_ONLY,
        "nvjpeg_hw_decoder",
        id="hw",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_roi_subsampled_rejected_when_disabled(backend_kind, decoder_id):
    """With enable_roi_fancy_upsampling=0, ROI decode on a non-4:4:4 image must be
    rejected (decode returns None) when using the nvjpeg backend exclusively."""
    options = f"{decoder_id}:enable_roi_fancy_upsampling=0"
    dec = _make_decoder(backend_kind, options)
    result = _roi_decode(dec, _SUBSAMPLED_IMG)
    assert result is None, (
        f"Expected rejection (None) for ROI+subsampled with enable_roi_fancy_upsampling=0 "
        f"on {decoder_id}, but got an image of shape {result.shape if result is not None else 'N/A'}"
    )


@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_roi_444_not_rejected_when_disabled(backend_kind, decoder_id):
    """4:4:4 images have no chroma upsampling, so enable_roi_fancy_upsampling=0
    must not block their ROI decode."""
    options = f"{decoder_id}:enable_roi_fancy_upsampling=0"
    dec = _make_decoder(backend_kind, options)
    result = _roi_decode(dec, _444_IMG)
    assert result is not None, (
        f"ROI decode of a 4:4:4 image should succeed even with enable_roi_fancy_upsampling=0 "
        f"on {decoder_id}"
    )
    expected_h = _ROI.end[0] - _ROI.start[0]
    expected_w = _ROI.end[1] - _ROI.start[1]
    assert result.shape[:2] == (expected_h, expected_w)


@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_roi_gray_not_rejected_when_disabled(backend_kind, decoder_id):
    """Grayscale images have no chroma upsampling, so enable_roi_fancy_upsampling=0
    must not block their ROI decode."""
    options = f"{decoder_id}:enable_roi_fancy_upsampling=0"
    dec = _make_decoder(backend_kind, options)
    result = _roi_decode(dec, _GRAY_IMG)
    assert result is not None, (
        f"ROI decode of a grayscale image should succeed even with enable_roi_fancy_upsampling=0 "
        f"on {decoder_id}"
    )
    expected_h = _ROI.end[0] - _ROI.start[0]
    expected_w = _ROI.end[1] - _ROI.start[1]
    assert result.shape[:2] == (expected_h, expected_w)


@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_full_image_not_rejected_when_disabled(backend_kind, decoder_id):
    """Full-image decode (no ROI) must succeed regardless of enable_roi_fancy_upsampling."""
    options = f"{decoder_id}:enable_roi_fancy_upsampling=0"
    dec = _make_decoder(backend_kind, options)
    result = _full_decode(dec, _SUBSAMPLED_IMG)
    assert result is not None, (
        f"Full-image decode should succeed even with enable_roi_fancy_upsampling=0 "
        f"on {decoder_id}"
    )


@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_roi_subsampled_accepted_when_enabled(backend_kind, decoder_id):
    """With enable_roi_fancy_upsampling=1, ROI decode on a subsampled image must succeed."""
    options = f"{decoder_id}:enable_roi_fancy_upsampling=1"
    dec = _make_decoder(backend_kind, options)
    result = _roi_decode(dec, _SUBSAMPLED_IMG)
    assert result is not None, (
        f"ROI decode of a subsampled image should succeed with enable_roi_fancy_upsampling=1 "
        f"on {decoder_id}"
    )
    expected_h = _ROI.end[0] - _ROI.start[0]
    expected_w = _ROI.end[1] - _ROI.start[1]
    assert result.shape[:2] == (expected_h, expected_w)


@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_default_enabled_on_new_nvjpeg(backend_kind, decoder_id):
    """On nvJPEG >= 13.2 the default for enable_roi_fancy_upsampling is 1, so ROI
    decode on a subsampled image must succeed without any explicit option."""
    nvjpeg_ver = get_nvjpeg_ver()
    if nvjpeg_ver == (0, 0, 0) or nvjpeg_ver < NVJPEG_WITH_FIXED_UPSAMPLING_VERSION:
        pytest.skip(f"nvJPEG version {nvjpeg_ver} < 13.2; default is disabled on this platform")
    dec = _make_decoder(backend_kind)
    result = _roi_decode(dec, _SUBSAMPLED_IMG)
    assert result is not None, (
        f"ROI decode should succeed by default on nvJPEG >= 13.2 on {decoder_id}"
    )


@pytest.mark.parametrize("backend_kind,decoder_id", _BACKENDS)
def test_default_disabled_on_old_nvjpeg(backend_kind, decoder_id):
    """On nvJPEG < 13.2 the default for enable_roi_fancy_upsampling is 0, so ROI
    decode on a subsampled image must be rejected without any explicit option."""
    nvjpeg_ver = get_nvjpeg_ver()
    if nvjpeg_ver == (0, 0, 0) or nvjpeg_ver >= NVJPEG_WITH_FIXED_UPSAMPLING_VERSION:
        pytest.skip(f"nvJPEG version {nvjpeg_ver} >= 13.2; default is enabled on this platform")
    dec = _make_decoder(backend_kind)
    result = _roi_decode(dec, _SUBSAMPLED_IMG)
    assert result is None, (
        f"ROI decode on subsampled image should be rejected by default on nvJPEG < 13.2 "
        f"on {decoder_id}"
    )
