# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the ``DecodeParams.sample_format`` knob and the matching
``sample_format=`` argument on ``as_image`` / ``as_images``.

Verifies that:
  * Decoding produces both interleaved (I_*) and planar (P_*) layouts.
  * Channel order families honour color_spec.
  * Backward compatibility: omitting ``sample_format`` reproduces the legacy
    interleaved-by-default behaviour, including ``DecodeParams()``.
  * ``as_image`` accepts a CHW array when given a planar sample_format.
  * Invalid combinations raise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest as t

from nvidia import nvimgcodec

from utils import (
    compare_host_images,
    get_opencv_reference,
    img_dir_path,
)

try:
    import cupy as cp
    # Force CUDA + libnvrtc to load eagerly. cp.zeros is a plain allocation
    # that does not trigger kernel compilation, so libnvrtc errors slip
    # through and only surface mid-test; cp.random.randint compiles a
    # kernel, which catches missing libnvrtc.so.12 / libcuda.so here.
    cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8)
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False
    cp = None


# Each decode test runs twice: once pinned to CPU_ONLY, once with the
# library's default backend priority list (all backends enabled, GPU
# preferred). The "cpu" run forces the CPU fallbacks for every format.
BACKENDS = [
    ("cpu", [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]),
    ("all", None),
]


def _decoder(backends=None):
    """Build a Decoder for one of the BACKENDS entries.

    ``backends=None`` selects the library's default priority list (all
    backends enabled).
    """
    return nvimgcodec.Decoder(backends=backends)


def _decode(decoder, path, params, backend_name):
    """Decode ``path`` and fail the test with a clear message if the chosen
    backend produces no image. All formats listed in this file are expected
    to decode on every backend - a ``None`` result is a real failure, not a
    skip."""
    img = decoder.read(path, params=params)
    assert img is not None, (
        f"Backend '{backend_name}' returned None for "
        f"{os.path.relpath(path, img_dir_path)} (params={params!r}). "
        f"Decoding this input was expected to succeed.")
    return img


# ---------------------------------------------------------------------------
# Decode to a requested layout (planar vs interleaved)
# ---------------------------------------------------------------------------

# Format coverage matrix. The lists below name *every* on-disk format that
# the codec library currently supports decoding through the Python bindings.
# Each entry is paired with the codec family it goes through so that test ids
# stay readable.

_RGB_FILES = [
    # JPEG: source color_spec=SYCC, decoder converts to sRGB on decode.
    "jpeg/padlock-406986_640_444.jpg",
    "jpeg/padlock-406986_640_420.jpg",
    "png/cat-1245673_640.png",
    "jpeg2k/cat-1046544_640.jp2",
    "bmp/cat-111793_640.bmp",
    "tiff/cat-300572_640.tiff",
    # WebP: both lossy and lossless variants.
    "webp/lossy/cat-3113513_640.webp",
    "webp/lossless/cat-3113513_640.webp",
    "pnm/cat-111793_640.ppm",
]

_GRAY_FILES = [
    "jpeg/padlock-406986_640_gray.jpg",
    "jpeg2k/tiled-cat-1046544_640_gray.jp2",
    "tiff/cat-300572_640_grayscale.tiff",
    # PGM (single-channel PNM).
    "pnm/cat-1245673_640.pgm",
]

_RGBA_FILES = [
    # PNG with alpha.
    "png/with_alpha/cat-111793_640-alpha.png",
    # JPEG 2000 with alpha.
    "jpeg2k/with_alpha/cat-111793_640-alpha.jp2",
    # WebP lossless with alpha.
    "webp/lossless_alpha/camel-1987672_640.webp",
]


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _RGB_FILES)
@t.mark.parametrize(
    "sample_format,expected_layout,expected_shape_last_or_first",
    [
        (nvimgcodec.SampleFormat.I_RGB, "interleaved", 3),
        (nvimgcodec.SampleFormat.P_RGB, "planar", 3),
    ],
)
def test_decode_to_layout_rgb(input_img_file, sample_format, expected_layout,
                              expected_shape_last_or_first, backend_name, backends):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB, sample_format=sample_format)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)

    assert img.sample_format == sample_format
    assert img.color_spec == nvimgcodec.ColorSpec.SRGB
    if expected_layout == "interleaved":
        # Shape: (H, W, C)
        assert img.shape[2] == expected_shape_last_or_first
    else:
        # Shape: (C, H, W)
        assert img.shape[0] == expected_shape_last_or_first
        assert img.ndim == 3
    assert img.num_channels == 3

    # Sanity: pixel content equals OpenCV reference when transposed for planar.
    ref = get_opencv_reference(input_img_path)
    np_img = np.asarray(img.cpu())
    if expected_layout == "planar":
        np_img = np.transpose(np_img, (1, 2, 0))
    compare_host_images([np_img], [ref])


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _RGB_FILES)
@t.mark.parametrize(
    "sample_format,expected_layout",
    [
        (nvimgcodec.SampleFormat.I_BGR, "interleaved"),
        (nvimgcodec.SampleFormat.P_BGR, "planar"),
    ],
)
def test_decode_to_layout_bgr(input_img_file, sample_format, expected_layout,
                              backend_name, backends):
    """Mirror of the RGB layout test, but requesting BGR channel order.

    The decoder swaps R and B relative to the I_RGB / P_RGB output. To verify
    that, we compare against ``get_opencv_reference`` (which returns RGB)
    after swapping the BGR result back to RGB along the channel axis.
    """
    input_img_path = os.path.join(img_dir_path, input_img_file)
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB, sample_format=sample_format)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)

    assert img.sample_format == sample_format
    assert img.color_spec == nvimgcodec.ColorSpec.SRGB
    if expected_layout == "interleaved":
        assert img.shape[2] == 3
    else:
        assert img.shape[0] == 3
        assert img.ndim == 3
    assert img.num_channels == 3

    # Swap BGR -> RGB along the channel axis, then compare against the RGB
    # OpenCV reference. For planar, transpose CHW -> HWC first so the channel
    # axis is the last one.
    np_img = np.asarray(img.cpu())
    if expected_layout == "planar":
        np_img = np.transpose(np_img, (1, 2, 0))
    np_img_as_rgb = np_img[..., ::-1]
    ref_rgb = get_opencv_reference(input_img_path)
    compare_host_images([np_img_as_rgb], [ref_rgb])


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _GRAY_FILES)
@t.mark.parametrize(
    "sample_format,expected_layout",
    [
        (nvimgcodec.SampleFormat.I_Y, "interleaved"),
        (nvimgcodec.SampleFormat.P_Y, "planar"),
    ],
)
def test_decode_to_layout_gray(input_img_file, sample_format, expected_layout,
                               backend_name, backends):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.GRAY, sample_format=sample_format)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)

    assert img.sample_format == sample_format
    assert img.color_spec == nvimgcodec.ColorSpec.GRAY
    if expected_layout == "interleaved":
        assert img.shape[2] == 1
    else:
        assert img.shape[0] == 1
    assert img.num_channels == 1


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _RGBA_FILES)
@t.mark.parametrize(
    "sample_format,expected_layout",
    [
        (nvimgcodec.SampleFormat.I_RGBA, "interleaved"),
        (nvimgcodec.SampleFormat.P_RGBA, "planar"),
    ],
)
def test_decode_to_layout_rgba(input_img_file, sample_format, expected_layout,
                               backend_name, backends):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    # color_spec=UNCHANGED lets the source's 4 channels (with alpha) pass
    # through. color_spec=SRGB would force 3-channel output.
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, sample_format=sample_format)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)

    assert img.sample_format == sample_format
    if expected_layout == "interleaved":
        assert img.shape[2] == 4
    else:
        assert img.shape[0] == 4
    assert img.ndim == 3
    assert img.num_channels == 4


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize(
    "sample_format,expected_layout",
    [
        (nvimgcodec.SampleFormat.I_YUV, "interleaved"),
        (nvimgcodec.SampleFormat.P_YUV, "planar"),
    ],
)
def test_decode_jpeg_to_yuv_layout(sample_format, expected_layout, backend_name, backends):
    """JPEG with color_spec=SYCC + explicit I_YUV/P_YUV must skip the
    YCbCr->sRGB conversion and emit raw YCbCr samples in the requested layout."""
    input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_444.jpg")
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SYCC, sample_format=sample_format)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)

    assert img.sample_format == sample_format
    assert img.color_spec == nvimgcodec.ColorSpec.SYCC
    if expected_layout == "interleaved":
        assert img.shape[2] == 3
    else:
        assert img.shape[0] == 3
        assert img.ndim == 3
    assert img.num_channels == 3


# ---------------------------------------------------------------------------
# Backward compatibility: omitting sample_format reproduces today's behaviour
# ---------------------------------------------------------------------------

@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _RGB_FILES)
def test_decode_default_unchanged_is_interleaved(input_img_file, backend_name, backends):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    # Default DecodeParams: color_spec=SRGB, sample_format=I_UNCHANGED.
    img = _decode(_decoder(backends), input_img_path,
                          nvimgcodec.DecodeParams(), backend_name)
    assert img.sample_format == nvimgcodec.SampleFormat.I_RGB
    assert img.color_spec == nvimgcodec.ColorSpec.SRGB
    assert img.shape[2] == 3
    assert img.num_channels == 3


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize(
    "input_img_file",
    [
        # JPEG: source family is SYCC, gets promoted to SRGB on output.
        "jpeg/padlock-406986_640_444.jpg",
        "jpeg/padlock-406986_640_420.jpg",
    ],
)
def test_decode_unchanged_color_spec_on_sycc_source_labels_unchanged(input_img_file, backend_name, backends):
    """When the caller requests color_spec=UNCHANGED and the source's family
    is not a natural output family (e.g. SYCC for JPEG, CMYK, YCCK), the
    decoder converts to sRGB but labels the result as I_UNCHANGED /
    P_UNCHANGED to distinguish "decoded as-is" from "explicitly converted
    to I_RGB". Matches the legacy behaviour of color_spec=UNCHANGED on a
    YCC JPEG."""
    input_img_path = os.path.join(img_dir_path, input_img_file)
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)
    assert img.sample_format == nvimgcodec.SampleFormat.I_UNCHANGED
    assert img.color_spec == nvimgcodec.ColorSpec.SRGB
    assert img.shape[2] == 3

# ---------------------------------------------------------------------------
# Invalid combinations raise
# ---------------------------------------------------------------------------

@t.mark.parametrize(
    "color_spec,sample_format",
    [
        # SRGB with a non-RGB-family format
        (nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.I_YUV),
        (nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.P_Y),
        # GRAY with an RGB-family format
        (nvimgcodec.ColorSpec.GRAY, nvimgcodec.SampleFormat.I_RGB),
        # SYCC with an RGB-family format
        (nvimgcodec.ColorSpec.SYCC, nvimgcodec.SampleFormat.P_RGB),
    ],
)
def test_decode_incompatible_color_spec_and_sample_format_raises(
        color_spec, sample_format):
    # The resolver's compatibility check fires before any backend touches the
    # bitstream, so this raises identically on every backend. The same shape
    # of error must surface whether the params are built positionally via the
    # constructor or assembled via property setters.
    input_img_path = os.path.join(img_dir_path, _RGB_FILES[0])
    params = nvimgcodec.DecodeParams(color_spec=color_spec, sample_format=sample_format)
    with t.raises(ValueError):
        _decoder().read(input_img_path, params=params)

@t.mark.parametrize(
    "color_spec",
    [
        nvimgcodec.ColorSpec.PALETTE,
        nvimgcodec.ColorSpec.ICC_PROFILE,
        # UNSUPPORTED (-1) is the negative sentinel; it must be rejected too and
        # must not slip past the range guard.
        nvimgcodec.ColorSpec.UNSUPPORTED,
    ],
)
def test_decode_unsupported_color_spec_raises(color_spec):
    """``ColorSpec.PALETTE``, ``ColorSpec.ICC_PROFILE`` and ``ColorSpec.UNSUPPORTED``
    are valid enum values but the resolver refuses them as decoder requests - the
    resolver only knows how to drive natural output families."""
    input_img_path = os.path.join(img_dir_path, _RGB_FILES[0])
    params = nvimgcodec.DecodeParams(color_spec=color_spec)
    with t.raises(ValueError, match="unsupported color_spec"):
        _decoder().read(input_img_path, params=params)

@t.mark.parametrize(
    "sample_format",
    [
        # UNKNOWN (0) is a valid enum value but not a valid decoder request -
        # the user should use *_UNCHANGED instead.
        nvimgcodec.SampleFormat.UNKNOWN,
        # UNSUPPORTED (-1) is the negative sentinel; it must be rejected too and
        # must not slip through the range guard.
        nvimgcodec.SampleFormat.UNSUPPORTED,
    ],
)
def test_decode_unsupported_sample_format_raises(sample_format):
    input_img_path = os.path.join(img_dir_path, _RGB_FILES[0])
    params = nvimgcodec.DecodeParams(sample_format=sample_format)
    # The diagnostic must name sample_format (the field at fault), not color_spec.
    with t.raises(ValueError, match="unsupported sample_format"):
        _decoder().read(input_img_path, params=params)


@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize(
    "color_spec,sample_format",
    [
        # UNCHANGED color_spec accepts any sample_format.
        (nvimgcodec.ColorSpec.UNCHANGED, nvimgcodec.SampleFormat.I_RGB),
        (nvimgcodec.ColorSpec.UNCHANGED, nvimgcodec.SampleFormat.P_RGB),
        # UNCHANGED sample_format accepts any color_spec.
        (nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.I_UNCHANGED),
        (nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.P_UNCHANGED),
    ],
)
def test_decode_unchanged_combinations_are_accepted(
        color_spec, sample_format, backend_name, backends):
    input_img_path = os.path.join(img_dir_path, _RGB_FILES[0])
    params = nvimgcodec.DecodeParams(color_spec=color_spec, sample_format=sample_format)
    img = _decode(_decoder(backends), input_img_path, params, backend_name)
    assert img is not None


# ---------------------------------------------------------------------------
# Round-trip: decode planar, wrap back through as_image, metadata matches.
# ---------------------------------------------------------------------------

@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_decode_planar_then_as_image_round_trip(backend_name, backends):
    input_img_path = os.path.join(img_dir_path, _RGB_FILES[0])
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.P_RGB,
    )
    img = _decode(_decoder(backends), input_img_path, params, backend_name)
    arr = np.asarray(img.cpu())  # (3, H, W)
    assert arr.shape[0] == 3

    wrapped = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert wrapped.sample_format == nvimgcodec.SampleFormat.P_RGB
    assert wrapped.shape == img.shape


# ---------------------------------------------------------------------------
# Decode into a pre-allocated row-padded buffer (reuse path).
#
# The caller wraps a padded buffer with as_image and passes it as `image=` so
# the decoder writes into the existing memory instead of allocating its own.
# Pixel content must match a fresh decode of the same code stream.
#
# Parametrised over:
#   - backend          : cpu-only vs default backend list (BACKENDS above)
#   - array_module     : numpy (host buffer) vs cupy (device buffer)
#   - layout           : interleaved (HWC) vs planar (CHW)
#   - format           : a curated set of files covering JPEG / PNG / BMP /
#                        TIFF / WEBP / JPEG2K / PNM
# ---------------------------------------------------------------------------


_PADDED_DECODE_RGB_FILES = [
    "jpeg/padlock-406986_640_444.jpg",
    "png/cat-1245673_640.png",
    "bmp/cat-111793_640.bmp",
    "tiff/cat-300572_640.tiff",
    "webp/lossy/cat-3113513_640.webp",
    "webp/lossless/cat-3113513_640.webp",
    "jpeg2k/cat-1046544_640.jp2",
    "pnm/cat-111793_640.ppm",
]

_PADDED_DECODE_GRAY_FILES = [
    "jpeg/padlock-406986_640_gray.jpg",
    "pnm/cat-1245673_640.pgm",
    "tiff/cat-300572_640_grayscale.tiff",
    "jpeg2k/tiled-cat-1046544_640_gray.jp2",
]


_ARRAY_MODULES = [
    ("numpy", np),
    t.param("cupy", cp,
            marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
]


def _as_host(arr):
    """Bring an ndarray-like (numpy or cupy) to host."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# A non-zero sentinel pre-fills the entire backing buffer; after decode the
# padded region (everything outside the slice the Image was wrapped on) must
# still hold this value, proving the codec did not write past the visible
# bounds.
PADDING_SENTINEL = 0x80


def _padding_reference(am, padded_shape):
    """Reference padded array that is sentinel-valued everywhere - used to
    check, after decode, that the padded region of the real buffer is byte-
    for-byte identical."""
    return np.full(padded_shape, PADDING_SENTINEL, dtype=np.uint8)


@t.mark.parametrize("am_name,am", _ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _PADDED_DECODE_RGB_FILES)
def test_decode_into_padded_hwc_buffer(input_img_file, backend_name, backends, am_name, am):
    """Decode an RGB source into a padded externally-managed HWC buffer.

    Padded shape `(H, W + pad, 3)` is sliced to `(H, W, 3)`; the row stride
    on the resulting Image is `(W + pad) * 3`. The decoded pixels must match
    a fresh decode, and the padded columns must remain at the sentinel value
    they were pre-filled with (no writes outside the visible region).
    """
    input_img_path = os.path.join(img_dir_path, input_img_file)
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 32
    padded = am.full((H, W + pad, 3), PADDING_SENTINEL, dtype=am.uint8)
    view = padded[:, :W, :]
    target = nvimgcodec.as_image(view)
    assert target.shape == (H, W, 3)
    assert target.strides[0] == (W + pad) * 3

    decode_params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB)
    decoder = _decoder(backends)
    decoded = _decode(decoder, input_img_path, decode_params, backend_name)
    decoder_for_target = _decoder(backends)
    written = decoder_for_target.decode(cs, params=decode_params, image=target)

    assert written.shape == (H, W, 3)
    ref_host = _as_host(decoded.cpu())
    # 1. The Image returned by decode matches a fresh reference decode.
    np.testing.assert_array_equal(_as_host(written.cpu()), ref_host)
    # 2. The underlying input array's visible slice reflects the decoded pixels.
    np.testing.assert_array_equal(_as_host(view), ref_host)
    # 3. The padded columns are byte-for-byte untouched (still sentinel).
    np.testing.assert_array_equal(
        _as_host(padded[:, W:, :]),
        _padding_reference(am, (H, pad, 3)),
    )


@t.mark.parametrize("am_name,am", _ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _PADDED_DECODE_RGB_FILES)
def test_decode_into_padded_chw_buffer(input_img_file, backend_name, backends, am_name, am):
    """Decode an RGB source into a padded externally-managed planar CHW buffer.

    Padded shape `(3, H, W + pad)` sliced to `(3, H, W)` - row stride is the
    padded width, plane stride is `H * (W + pad)` (no plane gap). Decoder
    must write into the buffer respecting the padded row stride for every
    plane, and the per-row tail bytes must remain at the sentinel.
    """
    input_img_path = os.path.join(img_dir_path, input_img_file)
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 32
    padded = am.full((3, H, W + pad), PADDING_SENTINEL, dtype=am.uint8)
    view = padded[:, :, :W]
    target = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert target.shape == (3, H, W)
    assert target.strides[0] == H * (W + pad)
    assert target.strides[1] == (W + pad)

    decode_params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.P_RGB,
    )
    decoder = _decoder(backends)
    reference = _decode(decoder, input_img_path, decode_params, backend_name)
    decoder_for_target = _decoder(backends)
    written = decoder_for_target.decode(cs, params=decode_params, image=target)

    assert written.shape == (3, H, W)
    ref_host = _as_host(reference.cpu())
    # 1. Image returned by decode matches a fresh reference decode.
    np.testing.assert_array_equal(_as_host(written.cpu()), ref_host)
    # 2. Visible slice of the user-provided buffer reflects the decoded pixels.
    np.testing.assert_array_equal(_as_host(view), ref_host)
    # 3. Per-row tail bytes (one block per plane) are still sentinel.
    np.testing.assert_array_equal(
        _as_host(padded[:, :, W:]),
        _padding_reference(am, (3, H, pad)),
    )


@t.mark.parametrize("am_name,am", _ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _PADDED_DECODE_GRAY_FILES)
def test_decode_grayscale_into_padded_hwc_buffer(input_img_file, backend_name, backends, am_name, am):
    """Decode a grayscale source into a padded `(H, W+pad, 1)` HWC buffer."""
    input_img_path = os.path.join(img_dir_path, input_img_file)
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 16
    padded = am.full((H, W + pad, 1), PADDING_SENTINEL, dtype=am.uint8)
    view = padded[:, :W, :]
    target = nvimgcodec.as_image(view)
    assert target.shape == (H, W, 1)
    assert target.strides[0] == W + pad

    decode_params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY)
    decoder = _decoder(backends)
    reference = _decode(decoder, input_img_path, decode_params, backend_name)
    decoder_for_target = _decoder(backends)
    written = decoder_for_target.decode(cs, params=decode_params, image=target)

    assert written.shape == (H, W, 1)
    ref_host = _as_host(reference.cpu())
    np.testing.assert_array_equal(_as_host(written.cpu()), ref_host)
    np.testing.assert_array_equal(_as_host(view), ref_host)
    np.testing.assert_array_equal(
        _as_host(padded[:, W:, :]),
        _padding_reference(am, (H, pad, 1)),
    )


@t.mark.parametrize("am_name,am", _ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _PADDED_DECODE_GRAY_FILES)
def test_decode_grayscale_into_padded_chw_buffer(input_img_file, backend_name, backends, am_name, am):
    """Decode a grayscale source into a padded planar `(1, H, W+pad)` buffer."""
    input_img_path = os.path.join(img_dir_path, input_img_file)
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 16
    padded = am.full((1, H, W + pad), PADDING_SENTINEL, dtype=am.uint8)
    view = padded[:, :, :W]
    target = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert target.shape == (1, H, W)

    decode_params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.GRAY,
        sample_format=nvimgcodec.SampleFormat.P_Y,
    )
    decoder = _decoder(backends)
    reference = _decode(decoder, input_img_path, decode_params, backend_name)
    decoder_for_target = _decoder(backends)
    written = decoder_for_target.decode(cs, params=decode_params, image=target)

    assert written.shape == (1, H, W)
    ref_host = _as_host(reference.cpu())
    np.testing.assert_array_equal(_as_host(written.cpu()), ref_host)
    np.testing.assert_array_equal(_as_host(view), ref_host)
    np.testing.assert_array_equal(
        _as_host(padded[:, :, W:]),
        _padding_reference(am, (1, H, pad)),
    )


@t.mark.parametrize("am_name,am", _ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", BACKENDS, ids=[b[0] for b in BACKENDS])
@t.mark.parametrize("input_img_file", _PADDED_DECODE_GRAY_FILES)
def test_decode_grayscale_2d_reuse_buffer(input_img_file, backend_name, backends, am_name, am):
    """A 2-D (H, W+pad) buffer wraps as a single-channel I_Y Image and serves
    as a reuse target for a grayscale decode. Verifies the 2-D inference
    path against a fresh reference decode, that the underlying 2-D view
    holds the decoded pixels, and that the padded tail columns survive
    unwritten."""
    input_img_path = os.path.join(img_dir_path, input_img_file)
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 16
    padded_2d = am.full((H, W + pad), PADDING_SENTINEL, dtype=am.uint8)
    view_2d = padded_2d[:, :W]
    target = nvimgcodec.as_image(view_2d)
    # 2-D input is inferred as I_Y, shape (H, W, 1).
    assert target.sample_format == nvimgcodec.SampleFormat.I_Y
    assert target.shape == (H, W, 1)

    decode_params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY)
    decoder = _decoder(backends)
    reference = _decode(decoder, input_img_path, decode_params, backend_name)
    decoder_for_target = _decoder(backends)
    written = decoder_for_target.decode(cs, params=decode_params, image=target)

    ref_host = _as_host(reference.cpu())  # (H, W, 1)
    np.testing.assert_array_equal(_as_host(written.cpu()), ref_host)
    # The 2-D slice's pixels match the (H, W, 1) reference squeezed to (H, W).
    np.testing.assert_array_equal(_as_host(view_2d), ref_host[..., 0])
    # Padded tail columns remain sentinel.
    np.testing.assert_array_equal(
        _as_host(padded_2d[:, W:]),
        _padding_reference(am, (H, pad)),
    )


# ---------------------------------------------------------------------------
# Cross-layout reuse
#
# A contiguous (no padding) buffer can be reused as a decoder target even
# when the decoder's resolved layout has a different plane count from the
# wrapped Image - the underlying bytes are reinterpreted under the new
# layout. A buffer with row or plane padding cannot be reused this way,
# because the padding gaps are not representable in the new layout.
# ---------------------------------------------------------------------------

def test_decode_planar_buffer_reused_for_interleaved_decode_contiguous():
    """User wraps a contiguous planar (3, H, W) buffer and asks the decoder
    to produce I_RGB. The bytes are reinterpreted as HWC and the decoder
    writes interleaved pixels into the same memory."""
    input_img_path = os.path.join(img_dir_path, _PADDED_DECODE_RGB_FILES[0])
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    backing = np.zeros((3, H, W), np.uint8)
    target = nvimgcodec.as_image(backing, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert target.shape == (3, H, W)

    decode_params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.I_RGB,
    )
    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(cs, params=decode_params, image=target)
    assert decoded is not None
    assert decoded.shape == (H, W, 3)

    reference = decoder.decode(
        cs, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB))
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), np.asarray(reference.cpu()))


def test_decode_interleaved_buffer_reused_for_planar_decode_contiguous():
    """User wraps a contiguous interleaved (H, W, 3) buffer and asks for
    P_RGB output. The bytes are reinterpreted as CHW."""
    input_img_path = os.path.join(img_dir_path, _PADDED_DECODE_RGB_FILES[0])
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    backing = np.zeros((H, W, 3), np.uint8)
    target = nvimgcodec.as_image(backing)
    assert target.shape == (H, W, 3)

    decode_params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.P_RGB,
    )
    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(cs, params=decode_params, image=target)
    assert decoded is not None
    assert decoded.shape == (3, H, W)

    reference = decoder.decode(
        cs, params=nvimgcodec.DecodeParams(
            color_spec=nvimgcodec.ColorSpec.SRGB,
            sample_format=nvimgcodec.SampleFormat.P_RGB))
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), np.asarray(reference.cpu()))


def test_decode_layout_mismatch_padded_planar_buffer_for_interleaved_decode_raises():
    """A padded planar buffer (per-row padding) cannot be cross-layout-reused:
    the padding bytes inside each row would have no place in the new HWC
    layout, so the resolver rejects the request rather than silently
    miswriting."""
    input_img_path = os.path.join(img_dir_path, _PADDED_DECODE_RGB_FILES[0])
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 32
    backing = np.zeros((3, H, W + pad), np.uint8)
    view = backing[:, :, :W]
    target = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert target.shape == (3, H, W)

    decode_params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.I_RGB,
    )
    decoder = nvimgcodec.Decoder()
    with t.raises((ValueError, RuntimeError),
                  match="Cross-layout reuse requires"):
        decoder.decode(cs, params=decode_params, image=target)


def test_decode_layout_mismatch_padded_interleaved_buffer_for_planar_decode_raises():
    """Mirror of the planar->interleaved padding rejection: a padded HWC
    buffer reused for a planar decode also raises."""
    input_img_path = os.path.join(img_dir_path, _PADDED_DECODE_RGB_FILES[0])
    cs = nvimgcodec.CodeStream(input_img_path)
    H, W = cs.height, cs.width

    pad = 32
    backing = np.zeros((H, W + pad, 3), np.uint8)
    view = backing[:, :W, :]
    target = nvimgcodec.as_image(view)
    assert target.shape == (H, W, 3)

    decode_params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.P_RGB,
    )
    decoder = nvimgcodec.Decoder()
    with t.raises((ValueError, RuntimeError),
                  match="Cross-layout reuse requires"):
        decoder.decode(cs, params=decode_params, image=target)
