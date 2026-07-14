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

"""Tests for multi-GPU device_id propagation and cross-device operations."""

from __future__ import annotations
import os
import threading

import numpy as np
import pytest as t

from nvidia import nvimgcodec

from utils import img_dir_path, compare_device_with_host_images, get_opencv_reference

try:
    from cuda.bindings import runtime as cudart
    err, count = cudart.cudaGetDeviceCount()
    print("cudaGetDeviceCount:", repr(err), getattr(err, "name", type(err)), "count=", count)
    num_devices = count if err == cudart.cudaError_t.cudaSuccess and count is not None else 0
except Exception:
    num_devices = 0

pytestmark = t.mark.skipif(num_devices < 2, reason=f"requires >= 2 GPUs, found {num_devices}")

JPEG_IMAGE = "jpeg/padlock-406986_640_420.jpg"
JP2_IMAGE = "jpeg2k/cat-1046544_640.jp2"
TIFF_IMAGE = "tiff/cat-300572_640.tiff"

IMAGES_BATCH = [
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg2k/cat-1046544_640.jp2",
    "bmp/cat-111793_640.bmp",
    "tiff/cat-300572_640.tiff",
]


def _all_device_ids():
    return list(range(num_devices))


def _pointer_device(ptr: int) -> int:
    assert ptr, "CUDA array interface returned a null data pointer"
    err, attrs = cudart.cudaPointerGetAttributes(ptr)
    assert err == cudart.cudaError_t.cudaSuccess, f"cudaPointerGetAttributes({ptr}) failed: {err}"
    return int(attrs.device)


def _assert_cuda_buffer_on_device(img, device_id: int):
    """Verify both nvImageCodec metadata and actual CUDA allocation placement."""
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    assert img.device_id == device_id
    ptr = int(img.__cuda_array_interface__["data"][0])
    actual_device = _pointer_device(ptr)
    assert actual_device == device_id, f"Image pointer {ptr} is on device {actual_device}, expected {device_id}"


# ---------------------------------------------------------------------------
# Decoder with device_id
# ---------------------------------------------------------------------------

@t.mark.parametrize("device_id", _all_device_ids())
def test_decode_single_on_each_gpu(device_id):
    """Decoder(device_id=N) should decode an image with the buffer on device N."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    assert decoder.device_id == device_id

    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    test_img = decoder.read(img_path)

    assert test_img is not None
    _assert_cuda_buffer_on_device(test_img, device_id)

    ref_img = get_opencv_reference(img_path)
    compare_device_with_host_images([test_img], [ref_img])


@t.mark.parametrize("device_id", _all_device_ids())
def test_decode_batch_on_each_gpu(device_id):
    """Batch decode should work on every GPU."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    img_paths = [os.path.join(img_dir_path, f) for f in IMAGES_BATCH]
    code_streams = [nvimgcodec.CodeStream(p) for p in img_paths]
    test_images = decoder.read(code_streams)

    assert len(test_images) == len(IMAGES_BATCH)
    for img in test_images:
        _assert_cuda_buffer_on_device(img, device_id)
    ref_images = [get_opencv_reference(p) for p in img_paths]
    compare_device_with_host_images(test_images, ref_images)


@t.mark.parametrize("device_id", _all_device_ids())
def test_decode_from_bytes_on_each_gpu(device_id):
    """Decoding from in-memory bytes should work on every GPU."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    img_path = os.path.join(img_dir_path, JPEG_IMAGE)

    with open(img_path, "rb") as f:
        data = f.read()
    test_img = decoder.decode(data)

    assert test_img is not None
    _assert_cuda_buffer_on_device(test_img, device_id)
    ref_img = get_opencv_reference(img_path)
    compare_device_with_host_images([test_img], [ref_img])


@t.mark.parametrize("device_id", _all_device_ids())
def test_decode_batch_from_bytes_on_each_gpu(device_id):
    """Batch decode from bytes should work on every GPU."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    img_paths = [os.path.join(img_dir_path, f) for f in IMAGES_BATCH]

    code_streams = []
    for p in img_paths:
        with open(p, "rb") as f:
            code_streams.append(nvimgcodec.CodeStream(f.read()))
    test_images = decoder.decode(code_streams)

    assert len(test_images) == len(IMAGES_BATCH)
    for img in test_images:
        _assert_cuda_buffer_on_device(img, device_id)
    ref_images = [get_opencv_reference(p) for p in img_paths]
    compare_device_with_host_images(test_images, ref_images)


# ---------------------------------------------------------------------------
# device_id property
# ---------------------------------------------------------------------------

@t.mark.parametrize("device_id", _all_device_ids())
def test_device_id_property(device_id):
    """Decoder, Encoder, and decoded Image should all expose the correct device_id."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    encoder = nvimgcodec.Encoder(device_id=device_id)

    assert decoder.device_id == device_id
    assert encoder.device_id == device_id

    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    img = decoder.read(nvimgcodec.CodeStream(img_path))
    _assert_cuda_buffer_on_device(img, device_id)


def test_default_device_id_resolves_to_current():
    """When device_id is omitted (default -1), it should resolve to the current CUDA device."""
    from cuda.bindings import runtime as cudart

    for dev in range(num_devices):
        cudart.cudaSetDevice(dev)

        decoder = nvimgcodec.Decoder()
        encoder = nvimgcodec.Encoder()
        assert decoder.device_id == dev, f"Decoder default device_id should be {dev}, got {decoder.device_id}"
        assert encoder.device_id == dev, f"Encoder default device_id should be {dev}, got {encoder.device_id}"

        img_path = os.path.join(img_dir_path, JPEG_IMAGE)
        img = decoder.read(nvimgcodec.CodeStream(img_path))
        _assert_cuda_buffer_on_device(img, dev)

    cudart.cudaSetDevice(0)


# ---------------------------------------------------------------------------
# Encoder with device_id
# ---------------------------------------------------------------------------

@t.mark.parametrize("device_id", _all_device_ids())
def test_encode_on_each_gpu(device_id):
    """Encoder(device_id=N) should encode an image decoded on device N."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    encoder = nvimgcodec.Encoder(device_id=device_id)
    assert encoder.device_id == device_id

    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    test_img = decoder.read(img_path)
    assert test_img is not None
    _assert_cuda_buffer_on_device(test_img, device_id)

    encoded = encoder.encode(test_img, codec="jpeg")
    assert encoded is not None


@t.mark.parametrize("device_id", _all_device_ids())
def test_encode_write_on_each_gpu(tmp_path, device_id):
    """Encode and write to file should work on every GPU."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    encoder = nvimgcodec.Encoder(device_id=device_id)

    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    test_img = decoder.read(img_path)
    assert test_img is not None
    _assert_cuda_buffer_on_device(test_img, device_id)

    out_path = str(tmp_path / f"output_gpu{device_id}.jpg")
    result = encoder.write(out_path, test_img)
    assert result is not None
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


# ---------------------------------------------------------------------------
# Round-trip (decode + encode + decode) on each GPU
# ---------------------------------------------------------------------------

@t.mark.parametrize("device_id", _all_device_ids())
def test_roundtrip_jp2_on_each_gpu(tmp_path, device_id):
    """Full decode -> encode -> decode round-trip on each GPU."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    encoder = nvimgcodec.Encoder(device_id=device_id)

    img_path = os.path.join(img_dir_path, JP2_IMAGE)
    original = decoder.read(img_path)
    assert original is not None
    _assert_cuda_buffer_on_device(original, device_id)

    params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    out_path = str(tmp_path / f"roundtrip_gpu{device_id}.jp2")
    encoder.write(out_path, original, params=params)
    assert os.path.exists(out_path)

    reloaded = decoder.read(out_path)
    assert reloaded is not None
    _assert_cuda_buffer_on_device(reloaded, device_id)

    np.testing.assert_array_equal(
        np.asarray(original.cpu()),
        np.asarray(reloaded.cpu()),
    )


# ---------------------------------------------------------------------------
# Image.cuda(device_id=...) cross-device transfer
# ---------------------------------------------------------------------------

def test_image_cuda_to_specific_device():
    """Image.cuda(device_id=N) should copy the image to device N."""
    ref = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    for dev in range(num_devices):
        device_img = host_img.cuda(device_id=dev)
        assert device_img is not None
        _assert_cuda_buffer_on_device(device_img, dev)
        np.testing.assert_array_equal(np.asarray(device_img.cpu()), ref)


def test_image_cuda_cross_device_via_host():
    """Decode on device 0, transfer to host, then to device 1, verify content."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    img_dev0 = decoder.read(img_path)
    assert img_dev0 is not None
    _assert_cuda_buffer_on_device(img_dev0, 0)

    img_dev1 = img_dev0.cpu().cuda(device_id=1)
    assert img_dev1 is not None
    _assert_cuda_buffer_on_device(img_dev1, 1)

    np.testing.assert_array_equal(
        np.asarray(img_dev0.cpu()),
        np.asarray(img_dev1.cpu()),
    )


def test_image_cuda_direct_device_to_device():
    """Decode on device 0, transfer directly to device 1 (GPU-to-GPU), verify content."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    img_dev0 = decoder.read(img_path)
    assert img_dev0 is not None
    _assert_cuda_buffer_on_device(img_dev0, 0)

    img_dev1 = img_dev0.cuda(device_id=1)
    assert img_dev1 is not None
    _assert_cuda_buffer_on_device(img_dev1, 1)
    assert img_dev1 is not img_dev0

    np.testing.assert_array_equal(
        np.asarray(img_dev0.cpu()),
        np.asarray(img_dev1.cpu()),
    )


def test_image_cuda_all_device_pairs():
    """Direct GPU-to-GPU copy between every pair of devices."""
    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    for src in range(num_devices):
        img_src = host_img.cuda(device_id=src)
        _assert_cuda_buffer_on_device(img_src, src)
        for dst in range(num_devices):
            img_dst = img_src.cuda(device_id=dst)
            assert img_dst is not None
            _assert_cuda_buffer_on_device(img_dst, dst)
            if src == dst:
                assert img_dst is img_src
            else:
                assert img_dst is not img_src
            np.testing.assert_array_equal(np.asarray(img_dst.cpu()), ref)


@t.mark.parametrize("image_file", [JPEG_IMAGE, JP2_IMAGE, TIFF_IMAGE])
def test_image_cuda_device_to_device_different_codecs(image_file):
    """Direct GPU-to-GPU copy works for images decoded from different codecs."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_path = os.path.join(img_dir_path, image_file)
    img_dev0 = decoder.read(img_path)
    assert img_dev0 is not None
    _assert_cuda_buffer_on_device(img_dev0, 0)

    img_dev1 = img_dev0.cuda(device_id=1)
    assert img_dev1 is not img_dev0
    _assert_cuda_buffer_on_device(img_dev1, 1)
    np.testing.assert_array_equal(
        np.asarray(img_dev0.cpu()),
        np.asarray(img_dev1.cpu()),
    )


def test_image_cuda_device_to_device_encode_on_target():
    """Decode on device 0, transfer to device 1, encode on device 1."""
    decoder = nvimgcodec.Decoder(device_id=0)
    encoder = nvimgcodec.Encoder(device_id=1)

    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    img_dev0 = decoder.read(img_path)
    _assert_cuda_buffer_on_device(img_dev0, 0)
    img_dev1 = img_dev0.cuda(device_id=1)
    _assert_cuda_buffer_on_device(img_dev1, 1)

    encoded = encoder.encode(img_dev1, codec="jpeg")
    assert encoded is not None


# ---------------------------------------------------------------------------
# Image.cuda() with cuda_stream parameter
# ---------------------------------------------------------------------------

def test_image_cuda_cross_device_with_cuda_stream():
    """Cross-device copy with an explicit cuda_stream on the target device."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    img_dev0 = decoder.read(nvimgcodec.CodeStream(img_path))
    _assert_cuda_buffer_on_device(img_dev0, 0)
    ref = np.asarray(img_dev0.cpu())

    stream = None
    try:
        cudart.cudaSetDevice(1)
        err, stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess
        img_dev1 = img_dev0.cuda(device_id=1, cuda_stream=stream)
        assert img_dev1 is not img_dev0
        _assert_cuda_buffer_on_device(img_dev1, 1)
        np.testing.assert_array_equal(np.asarray(img_dev1.cpu()), ref)
    finally:
        # Image.cuda(cuda_stream=s) detaches the new Image's buffer from `s`
        # before returning, so destroying the stream here is safe even while
        # img_dev1 is still alive.
        if stream is not None:
            cudart.cudaStreamDestroy(stream)
        cudart.cudaSetDevice(0)


def test_image_cuda_host_to_device_with_cuda_stream():
    """Host-to-device copy with an explicit cuda_stream."""
    ref = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    for dev in range(num_devices):
        cudart.cudaSetDevice(dev)
        err, stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess
        try:
            device_img = host_img.cuda(device_id=dev, cuda_stream=stream)
            assert device_img is not None
            _assert_cuda_buffer_on_device(device_img, dev)
            np.testing.assert_array_equal(np.asarray(device_img.cpu()), ref)
        finally:
            # Safe to destroy the stream immediately: Image.cuda detaches the
            # buffer from the stream before returning.
            cudart.cudaStreamDestroy(stream)

    cudart.cudaSetDevice(0)


def test_image_cuda_d2d_stream_lifetime_contract():
    """Regression: destroying the user's stream before the resulting Image is
    released must not terminate the process (device-to-device path).

    The library contract is scoped to the Image's buffer deallocation:
    after Image.cuda returns, Image destruction will not touch `s`. Other
    Image methods that enqueue work on the stream (e.g. Image.cpu) must
    still be called while `s` is alive — they read from
    image_info.cuda_stream. This test exercises the D2D (cross-device)
    async branch with `synchronize=False`, which is the exact hazard path
    the library-side fix addresses.

    The H2D (STRIDED_HOST -> STRIDED_DEVICE) counterpart lives in test_image.py.
    """
    decoder = nvimgcodec.Decoder(device_id=0)
    img_dev0 = decoder.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_IMAGE)))
    _assert_cuda_buffer_on_device(img_dev0, 0)

    cudart.cudaSetDevice(1)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess

    img_dev1 = img_dev0.cuda(device_id=1, cuda_stream=stream, synchronize=False)
    cudart.cudaStreamSynchronize(stream)
    _assert_cuda_buffer_on_device(img_dev1, 1)
    np.testing.assert_array_equal(np.asarray(img_dev1.cpu()), np.asarray(img_dev0.cpu()))

    cudart.cudaStreamDestroy(stream)
    del img_dev1

    cudart.cudaSetDevice(0)


def test_image_cuda_h2d_stream_device_mismatch_raises():
    """H2D path: cuda_stream on device 0 with device_id=1 must raise.

    `cudaMallocAsync` uses the memory pool of the stream's device, so a
    mismatched stream would silently allocate on the wrong device and leave
    the resulting Image.device_id inconsistent with the actual buffer.
    """
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        with t.raises(ValueError):
            host_img.cuda(device_id=1, cuda_stream=int(stream))
    finally:
        cudart.cudaStreamDestroy(stream)


def test_image_cuda_peer_stream_device_mismatch_raises():
    """Peer-copy path: same mismatch must raise before the peer copy is enqueued."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_dev0 = decoder.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_IMAGE)))
    _assert_cuda_buffer_on_device(img_dev0, 0)

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        with t.raises(ValueError):
            img_dev0.cuda(device_id=1, cuda_stream=int(stream))
    finally:
        cudart.cudaStreamDestroy(stream)


# cudaStreamLegacy == 0x1, cudaStreamPerThread == 0x2 are per-current-device
# sentinels. They must not trip the stream/device mismatch check even when the
# current device differs from the requested device_id, because they are
# resolved against the surrounding DeviceGuard.
@t.mark.parametrize("sentinel_stream", [1, 2])
def test_image_cuda_h2d_sentinel_stream_does_not_raise(sentinel_stream):
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    cudart.cudaSetDevice(0)
    try:
        device_img = host_img.cuda(device_id=1, cuda_stream=sentinel_stream)
        _assert_cuda_buffer_on_device(device_img, 1)
        np.testing.assert_array_equal(np.asarray(device_img.cpu()), ref)
    finally:
        cudart.cudaSetDevice(0)


# ---------------------------------------------------------------------------
# Image.cuda_stream property (cross-device cases)
# ---------------------------------------------------------------------------

def test_image_cuda_stream_property_cross_device():
    """Cross-device copy with explicit stream should report that stream on the new image."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_dev0 = decoder.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_IMAGE)))
    _assert_cuda_buffer_on_device(img_dev0, 0)

    stream = None
    try:
        cudart.cudaSetDevice(1)
        err, stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess
        stream_int = int(stream)
        img_dev1 = img_dev0.cuda(device_id=1, cuda_stream=stream_int)
        assert int(img_dev1.cuda_stream) == stream_int
        _assert_cuda_buffer_on_device(img_dev1, 1)
    finally:
        if stream is not None:
            cudart.cudaStreamDestroy(stream)
        cudart.cudaSetDevice(0)


# ---------------------------------------------------------------------------
# Image.device_id property (multi-GPU cases)
# ---------------------------------------------------------------------------

@t.mark.parametrize("device_id", _all_device_ids())
def test_image_device_id_property_on_each_gpu(device_id):
    """Images on GPU N should report device_id == N."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    img = decoder.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_IMAGE)))
    _assert_cuda_buffer_on_device(img, device_id)


def test_image_device_id_property_after_cuda():
    """host.cuda(device_id=1) should flip device_id from None to 1."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)
    assert host_img.device_id is None

    device_img = host_img.cuda(device_id=1)
    _assert_cuda_buffer_on_device(device_img, 1)


# ---------------------------------------------------------------------------
# __cuda_array_interface__ source-device detection
# ---------------------------------------------------------------------------

def test_image_from_cuda_array_interface_on_non_current_device():
    """as_image() on a CuPy array must detect the buffer's device via
    cudaPointerGetAttributes, not just fall back to cudaGetDevice."""
    cp = t.importorskip("cupy")

    with cp.cuda.Device(1):
        cp_arr = cp.asarray(np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8))
    cudart.cudaSetDevice(0)

    img = nvimgcodec.as_image(cp_arr)
    _assert_cuda_buffer_on_device(img, 1)

    np.testing.assert_array_equal(np.asarray(img.cpu()), cp.asnumpy(cp_arr))


def test_image_from_cuda_array_interface_all_devices():
    """as_image() correctly detects every device for CuPy-originated buffers."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    for dev in range(num_devices):
        with cp.cuda.Device(dev):
            cp_arr = cp.asarray(ref)

        cudart.cudaSetDevice((dev + 1) % num_devices)
        img = nvimgcodec.as_image(cp_arr)
        _assert_cuda_buffer_on_device(img, dev)
        np.testing.assert_array_equal(np.asarray(img.cpu()), ref)

    cudart.cudaSetDevice(0)


def test_image_cuda_cross_device_from_cuda_array_interface():
    """CuPy array on device 1, as_image(), then .cuda(device_id=0) should work."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    with cp.cuda.Device(1):
        cp_arr = cp.asarray(ref)

    img_dev1 = nvimgcodec.as_image(cp_arr)
    _assert_cuda_buffer_on_device(img_dev1, 1)

    img_dev0 = img_dev1.cuda(device_id=0)
    _assert_cuda_buffer_on_device(img_dev0, 0)
    assert img_dev0 is not img_dev1
    np.testing.assert_array_equal(np.asarray(img_dev0.cpu()), ref)


def test_image_cuda_cross_device_from_cuda_array_interface_with_stream():
    """CuPy array on device 1, as_image(), then .cuda(device_id=0, cuda_stream=stream_on_0).

    Exercises both the CAI source-device detection (cudaPointerGetAttributes) and the
    stream-scoped cudaMemcpyPeerAsync path together."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    with cp.cuda.Device(1):
        cp_arr = cp.asarray(ref)

    img_dev1 = nvimgcodec.as_image(cp_arr)
    _assert_cuda_buffer_on_device(img_dev1, 1)

    stream = None
    try:
        cudart.cudaSetDevice(0)
        err, stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

        img_dev0 = img_dev1.cuda(device_id=0, cuda_stream=stream)
        assert img_dev0 is not img_dev1
        _assert_cuda_buffer_on_device(img_dev0, 0)
        np.testing.assert_array_equal(np.asarray(img_dev0.cpu()), ref)
    finally:
        if stream is not None:
            cudart.cudaStreamDestroy(stream)
        cudart.cudaSetDevice(0)


# ---------------------------------------------------------------------------
# DLPack source-device detection
# ---------------------------------------------------------------------------

def test_image_from_dlpack_on_non_current_device():
    """as_image() via DLPack capsule must detect the buffer's device correctly."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    with cp.cuda.Device(1):
        cp_arr = cp.asarray(ref)
    cudart.cudaSetDevice(0)

    capsule = cp_arr.toDlpack()
    img = nvimgcodec.as_image(capsule)
    _assert_cuda_buffer_on_device(img, 1)
    np.testing.assert_array_equal(np.asarray(img.cpu()), ref)


def test_image_from_dlpack_all_devices():
    """as_image() via DLPack correctly detects every device."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    for dev in range(num_devices):
        with cp.cuda.Device(dev):
            cp_arr = cp.asarray(ref)

        cudart.cudaSetDevice((dev + 1) % num_devices)
        capsule = cp_arr.toDlpack()
        img = nvimgcodec.as_image(capsule)
        _assert_cuda_buffer_on_device(img, dev)
        np.testing.assert_array_equal(np.asarray(img.cpu()), ref)

    cudart.cudaSetDevice(0)


def test_image_from_dlpack_cross_device_copy():
    """DLPack-wrapped buffer on device 1, then .cuda(device_id=0)."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    with cp.cuda.Device(1):
        cp_arr = cp.asarray(ref)

    capsule = cp_arr.toDlpack()
    img_dev1 = nvimgcodec.as_image(capsule)
    _assert_cuda_buffer_on_device(img_dev1, 1)

    img_dev0 = img_dev1.cuda(device_id=0)
    _assert_cuda_buffer_on_device(img_dev0, 0)
    assert img_dev0 is not img_dev1
    np.testing.assert_array_equal(np.asarray(img_dev0.cpu()), ref)


def test_image_cuda_cross_device_from_dlpack_with_stream():
    """DLPack buffer on device 1, as_image(), then .cuda(device_id=0, cuda_stream=stream_on_0).

    Exercises both the DLPack source-device detection and the stream-scoped
    cudaMemcpyPeerAsync path together."""
    cp = t.importorskip("cupy")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    with cp.cuda.Device(1):
        cp_arr = cp.asarray(ref)

    capsule = cp_arr.toDlpack()
    img_dev1 = nvimgcodec.as_image(capsule)
    _assert_cuda_buffer_on_device(img_dev1, 1)

    stream = None
    try:
        cudart.cudaSetDevice(0)
        err, stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

        img_dev0 = img_dev1.cuda(device_id=0, cuda_stream=stream)
        assert img_dev0 is not img_dev1
        _assert_cuda_buffer_on_device(img_dev0, 0)
        np.testing.assert_array_equal(np.asarray(img_dev0.cpu()), ref)
    finally:
        if stream is not None:
            cudart.cudaStreamDestroy(stream)
        cudart.cudaSetDevice(0)


def test_image_from_pytorch_dlpack_on_non_current_device():
    """as_image() via PyTorch DLPack capsule must detect the buffer's device."""
    torch = t.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        t.skip("PyTorch needs >= 2 CUDA devices")

    ref = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    tensor = torch.from_numpy(ref).cuda(1)
    cudart.cudaSetDevice(0)

    capsule = torch.utils.dlpack.to_dlpack(tensor)
    img = nvimgcodec.as_image(capsule)
    _assert_cuda_buffer_on_device(img, 1)
    np.testing.assert_array_equal(np.asarray(img.cpu()), ref)


# ---------------------------------------------------------------------------
# Concurrent multi-GPU decoding (threaded)
# ---------------------------------------------------------------------------

def test_concurrent_decode_on_all_gpus():
    """Each GPU decodes the same image concurrently in separate threads."""
    img_path = os.path.join(img_dir_path, JPEG_IMAGE)
    ref_img = get_opencv_reference(img_path)

    results = [None] * num_devices
    errors = [None] * num_devices

    def worker(dev_id):
        try:
            dec = nvimgcodec.Decoder(device_id=dev_id)
            img = dec.read(nvimgcodec.CodeStream(img_path))
            _assert_cuda_buffer_on_device(img, dev_id)
            results[dev_id] = np.asarray(img.cpu())
        except Exception as e:
            errors[dev_id] = e

    threads = [threading.Thread(target=worker, args=(d,)) for d in range(num_devices)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    for d in range(num_devices):
        assert errors[d] is None, f"GPU {d} error: {errors[d]}"
        assert results[d] is not None, f"GPU {d} returned None"
        diff = np.abs(results[d].astype(np.int32) - ref_img.astype(np.int32))
        assert diff.max() <= 6, f"GPU {d} image mismatch, max diff={diff.max()}"


def test_concurrent_decode_batch_on_all_gpus():
    """Each GPU decodes a batch of images concurrently."""
    img_paths = [os.path.join(img_dir_path, f) for f in IMAGES_BATCH]
    ref_images = [get_opencv_reference(p) for p in img_paths]

    results = [None] * num_devices
    errors = [None] * num_devices

    def worker(dev_id):
        try:
            dec = nvimgcodec.Decoder(device_id=dev_id)
            code_streams = [nvimgcodec.CodeStream(p) for p in img_paths]
            imgs = dec.read(code_streams)
            for img in imgs:
                _assert_cuda_buffer_on_device(img, dev_id)
            results[dev_id] = [np.asarray(img.cpu()) for img in imgs]
        except Exception as e:
            errors[dev_id] = e

    threads = [threading.Thread(target=worker, args=(d,)) for d in range(num_devices)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    for d in range(num_devices):
        assert errors[d] is None, f"GPU {d} error: {errors[d]}"
        assert len(results[d]) == len(IMAGES_BATCH)
        for i, (test, ref) in enumerate(zip(results[d], ref_images)):
            diff = np.abs(test.astype(np.int32) - ref.astype(np.int32))
            assert diff.max() <= 6, f"GPU {d}, image {i}: max diff={diff.max()}"


def test_concurrent_encode_decode_roundtrip_on_all_gpus():
    """Each GPU does a full encode/decode round-trip concurrently."""
    img_path = os.path.join(img_dir_path, JP2_IMAGE)

    results = [None] * num_devices
    errors = [None] * num_devices

    def worker(dev_id, tmp_dir):
        try:
            dec = nvimgcodec.Decoder(device_id=dev_id)
            enc = nvimgcodec.Encoder(device_id=dev_id)

            original = dec.read(nvimgcodec.CodeStream(img_path))
            _assert_cuda_buffer_on_device(original, dev_id)
            params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
            out_path = os.path.join(tmp_dir, f"gpu{dev_id}.jp2")
            enc.write(out_path, original, params=params)

            reloaded = dec.read(nvimgcodec.CodeStream(out_path))
            _assert_cuda_buffer_on_device(reloaded, dev_id)
            results[dev_id] = (np.asarray(original.cpu()), np.asarray(reloaded.cpu()))
        except Exception as e:
            errors[dev_id] = e

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        threads = [threading.Thread(target=worker, args=(d, tmp_dir)) for d in range(num_devices)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    for d in range(num_devices):
        assert errors[d] is None, f"GPU {d} error: {errors[d]}"
        original, reloaded = results[d]
        np.testing.assert_array_equal(original, reloaded,
                                      err_msg=f"GPU {d}: lossless round-trip mismatch")

# ---------------------------------------------------------------------------
# Batch decode with CodeStream objects (explicit)
# ---------------------------------------------------------------------------

@t.mark.parametrize("device_id", _all_device_ids())
def test_batch_read_with_codestreams_on_each_gpu(device_id):
    """decoder.read(list[CodeStream]) should work on every GPU."""
    decoder = nvimgcodec.Decoder(device_id=device_id)
    img_paths = [os.path.join(img_dir_path, f) for f in IMAGES_BATCH]
    code_streams = [nvimgcodec.CodeStream(p) for p in img_paths]

    test_images = decoder.read(code_streams)

    assert len(test_images) == len(IMAGES_BATCH)
    for img in test_images:
        _assert_cuda_buffer_on_device(img, device_id)
    ref_images = [get_opencv_reference(p) for p in img_paths]
    compare_device_with_host_images(test_images, ref_images)
