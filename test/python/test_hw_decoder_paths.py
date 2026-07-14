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

"""
Tests that HW JPEG decode entry points produce identical pixel output.

The hw_decoder.cpp C++ layer selects one of two batched paths at runtime:
  1. Legacy batched path (nvjpegDecodeBatched*) - used when
     max_num_cpu_threads == 1, force_legacy_batch=1, or the batch heuristic
     declines BatchSingle.
  2. BatchSingle path (decodeBatchSingle) - per-image 3-function pipeline
     used by decodeBatch() when force_batch_single=1 or when the batch
     heuristic selects it.

The per-sample decode() API is routed through the same C++ path chooser with
batch_size=1. This file compares legacy batched decode, the cpp-threaded
batched candidate that hits BatchSingle on mixed-CSS multi-engine GPUs, a
forced BatchSingle decode, and Python-threaded single-image decode against the
same images.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import numpy as np
import pytest as t

from nvidia import nvimgcodec

from utils import img_dir_path


# Seed images covering a range of chroma subsampling modes (420, 422, 444,
# 440) and an EXIF-rotated image to exercise the orientation-correction path.
HW_COMPARE_SEED_IMAGES = [
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_444.jpg",
    "jpeg/padlock-406986_640_440.jpg",
    "jpeg/exif/padlock-406986_640_rotate_90.jpg",
]

# Decode one mixed-CSS batch of 15 images and repeat it 4 times for each
# entry point. With fancy upsampling enabled, batch_size >= 13 and multiple
# chroma subsampling values put the cpp-threaded decoder in the BatchSingle
# heuristic region on multi-engine HW decoder GPUs.
HW_COMPARE_BATCH_SIZE = 15
HW_COMPARE_ITERATIONS = 4

# Odd mixed-CSS batch used to exercise the BatchSingle heuristic with a
# non-seed-multiple size while keeping the path-comparison test inexpensive.
HW_TAIL_BATCH_SIZE = 17

# Number of concurrent Python-side decoder threads used in the
# single_python_threads path.  Tasks are queued, so this only needs to be >1 to
# exercise true Python-thread concurrency.
SINGLE_PYTHON_WORKERS = 4


def _make_hw_decoder(hw_decode_api: str, cpp_threads: int) -> nvimgcodec.Decoder:
    """Create an HW-only decoder configured for the requested decode entry point.

    hw_decode_api configures the decoder entry point:
      "batched"                  -> max_num_cpu_threads=1  (legacy nvjpegDecodeBatched* path)
      "batch_single_cpp_threads" -> max_num_cpu_threads=N  (BatchSingle heuristic candidate)
      "force_batch_single"       -> max_num_cpu_threads=N + force_batch_single=1
      "single_python_threads"    -> max_num_cpu_threads=1  (one-image decode calls, concurrency in Python)
    """
    if hw_decode_api in ("batched", "single_python_threads"):
        effective_threads = 1
    elif hw_decode_api in ("batch_single_cpp_threads", "force_batch_single"):
        effective_threads = cpp_threads
    else:
        raise ValueError(f"Unexpected hw_decode_api: {hw_decode_api}")

    options = ""
    if hw_decode_api == "force_batch_single":
        options += "nvjpeg_hw_decoder:force_batch_single=1"

    kwargs = {
        "backends": [nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY)],
        "options": options,
    }
    if effective_threads > 0:
        kwargs["max_num_cpu_threads"] = effective_threads
    return nvimgcodec.Decoder(**kwargs)


def _is_hw_decoder_unavailable(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "Could not create decoder" in message
        or "Requested backends not among available processors" in message
        or "nvJPEG HW decoder" in message
    )


def _make_codestream_batch(image_paths: list[str], input_mode: str) -> list[nvimgcodec.CodeStream]:
    """Build a list of CodeStream objects from file paths.

    input_mode="path"   passes the file path string directly (zero-copy I/O).
    input_mode="python" reads the bytes into Python memory first (in-memory I/O).
    """
    if input_mode == "path":
        return [nvimgcodec.CodeStream(path) for path in image_paths]
    if input_mode == "python":
        return [nvimgcodec.CodeStream(Path(path).read_bytes()) for path in image_paths]
    raise ValueError(f"Unexpected input_mode: {input_mode}")


def _make_codestream(path: str, input_mode: str) -> nvimgcodec.CodeStream:
    """Build a single CodeStream from a file path (see _make_codestream_batch)."""
    if input_mode == "path":
        return nvimgcodec.CodeStream(path)
    if input_mode == "python":
        return nvimgcodec.CodeStream(Path(path).read_bytes())
    raise ValueError(f"Unexpected input_mode: {input_mode}")


def _decode_batch_to_host(decoder: nvimgcodec.Decoder, image_paths: list[str], input_mode: str, params) -> list[np.ndarray]:
    """Decode a batch of images and return them as host (CPU) numpy arrays."""
    decoded = decoder.decode(_make_codestream_batch(image_paths, input_mode), params=params)
    assert len(decoded) == len(image_paths)

    host_images = []
    for path, image in zip(image_paths, decoded):
        assert image is not None, f"decode returned None for {path}"
        host_images.append(np.asarray(image.cpu()))
    return host_images


class _SinglePythonThreadExecutor:
    """Owns python_workers background threads.  Each thread holds its own
    nvimgcodec.Decoder (max_num_cpu_threads=1) and calls decode() for one
    image at a time.  From the C++ perspective this exercises the shared
    nvJPEG handle and worker pool under concurrent single-image load,
    because all Decoder objects on the same GPU share one SharedContext.

    Lifecycle:
      1. __init__: start threads and synchronise at ready_barrier.
      2. decode_batch_to_host: enqueue tasks, wait for completion, collect results.
      3. close: send stop sentinels and join all threads.
    """

    def __init__(self, input_mode: str, python_workers: int):
        self.input_mode = input_mode
        self.python_workers = python_workers
        self.stop_sentinel = object()
        self.task_queue: queue.Queue[object] = queue.Queue()
        # Barrier used to synchronise __init__ with threads after startup.
        self.ready_barrier = threading.Barrier(python_workers + 1)
        # Set once before the first decode_batch_to_host call; keeps threads
        # parked until the main thread is ready to enqueue real work.
        self.start_event = threading.Event()
        self.closed = False

        # One Decoder per thread; each uses max_num_cpu_threads=1. Each task is
        # submitted as a one-image decode() call and the C++ layer routes it
        # through the same batch_size=1 path chooser used by decodeBatch().
        self.decoders = [_make_hw_decoder("single_python_threads", cpp_threads=1) for _ in range(python_workers)]
        self.workers = [threading.Thread(target=self._worker, args=(tid,)) for tid in range(python_workers)]
        for worker_thread in self.workers:
            worker_thread.start()
        self.ready_barrier.wait()

    def _worker(self, tid: int) -> None:
        """Background thread body: wait for work, then drain the task queue until stopped."""
        decoder = self.decoders[tid]
        self.ready_barrier.wait()
        self.start_event.wait()   # block until decode_batch_to_host is called

        while True:
            task = self.task_queue.get()
            if task is self.stop_sentinel:
                self.task_queue.task_done()
                break

            batch_idx, path, params, batch_results = task
            try:
                decoded = decoder.decode(_make_codestream(path, self.input_mode), params=params)
                if decoded is None:
                    raise RuntimeError(f"decode returned None for {path}")
                batch_results[batch_idx] = np.asarray(decoded.cpu())
            except Exception as exc:
                # Store the exception; decode_batch_to_host will re-raise it.
                batch_results[batch_idx] = exc
            finally:
                self.task_queue.task_done()

    def decode_batch_to_host(self, image_paths: list[str], params) -> list[np.ndarray]:
        """Dispatch one decode task per image to the background threads and collect results."""
        self.start_event.set()  # no-op on subsequent calls

        batch_results: list[np.ndarray | Exception | None] = [None] * len(image_paths)
        for batch_idx, path in enumerate(image_paths):
            self.task_queue.put((batch_idx, path, params, batch_results))

        self.task_queue.join()  # block until all tasks are done

        host_images: list[np.ndarray] = []
        for path, result in zip(image_paths, batch_results):
            if isinstance(result, Exception):
                raise AssertionError(f"single_python_threads decode failed for {path}: {result}") from result
            assert result is not None, f"single_python_threads worker did not report a result for {path}"
            host_images.append(result)
        return host_images

    def close(self) -> None:
        """Stop all background threads gracefully."""
        if self.closed:
            return
        self.closed = True
        self.start_event.set()  # unblock threads that never received real work
        for _ in range(self.python_workers):
            self.task_queue.put(self.stop_sentinel)
        self.task_queue.join()
        for worker_thread in self.workers:
            worker_thread.join()


def _assert_same_outputs(
    reference_label: str,
    reference_images: list[np.ndarray],
    candidate_label: str,
    candidate_images: list[np.ndarray],
    image_paths: list[str],
) -> None:
    """Assert that two lists of decoded images are pixel-identical.

    Reports shape mismatches and the maximum absolute pixel difference on
    failure so the error message is actionable without re-running manually.
    """
    assert len(reference_images) == len(candidate_images) == len(image_paths)
    for index, (path, reference, candidate) in enumerate(zip(image_paths, reference_images, candidate_images)):
        assert reference.shape == candidate.shape, (
            f"shape mismatch for {candidate_label} vs {reference_label} on {path}: "
            f"{candidate.shape} != {reference.shape}"
        )
        if not np.array_equal(reference, candidate):
            diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
            raise AssertionError(
                f"pixel mismatch for {candidate_label} vs {reference_label} on {path} "
                f"at batch index {index}; max_abs_diff={int(diff.max())}"
            )


@t.mark.parametrize("input_mode", ["path", "python"])
def test_hw_decode_paths_match(input_mode: str):
    """Verify that HW JPEG decode entry points produce identical pixel output.

    Three decoders/executors are created:
      batched_decoder          -> max_num_cpu_threads=1, legacy nvjpegDecodeBatched* path
      batch_single_cpp_decoder -> max_num_cpu_threads=4; the mixed-CSS batch
                                  selects decodeBatchSingle on multi-engine GPUs
      single_python_executor   -> Python threads, each with max_num_cpu_threads=1

    The same 15-image batch is decoded 4 times through each entry point. Every
    iteration compares the candidate outputs against the legacy batched path.
    The test is skipped (not failed) when the HW decoder is unavailable on
    the current platform.
    """
    assert HW_COMPARE_BATCH_SIZE % len(HW_COMPARE_SEED_IMAGES) == 0
    repeats_per_batch = HW_COMPARE_BATCH_SIZE // len(HW_COMPARE_SEED_IMAGES)
    image_batch = [os.path.join(img_dir_path, rel_path) for rel_path in HW_COMPARE_SEED_IMAGES] * repeats_per_batch
    params = nvimgcodec.DecodeParams(apply_exif_orientation=True)

    try:
        batched_decoder = _make_hw_decoder("batched", cpp_threads=4)
        batch_single_cpp_decoder = _make_hw_decoder("batch_single_cpp_threads", cpp_threads=4)
        single_python_executor = _SinglePythonThreadExecutor(input_mode=input_mode, python_workers=SINGLE_PYTHON_WORKERS)
    except RuntimeError as exc:
        if _is_hw_decoder_unavailable(exc):
            t.skip(f"nvJPEG HW decoder not available: {exc}")
        raise

    try:
        for _ in range(HW_COMPARE_ITERATIONS):
            batched_images = _decode_batch_to_host(batched_decoder, image_batch, input_mode, params)
            batch_single_cpp_images = _decode_batch_to_host(batch_single_cpp_decoder, image_batch, input_mode, params)
            single_python_images = single_python_executor.decode_batch_to_host(image_batch, params)
            _assert_same_outputs("batched", batched_images, "batch_single_cpp_threads", batch_single_cpp_images, image_batch)
            _assert_same_outputs("batched", batched_images, "single_python_threads", single_python_images, image_batch)
    finally:
        single_python_executor.close()


@t.mark.parametrize("input_mode", ["path", "python"])
@t.mark.parametrize("cpp_threads", [2, 8])
def test_hw_batch_single_tail_matches_batched(input_mode: str, cpp_threads: int):
    """Compare the cpp-threaded BatchSingle candidate against legacy batched decode.

    The odd batch size keeps coverage distinct from the 15-image repeated batch
    and exercises a non-seed-multiple mixed-CSS batch. On single-engine HW
    decoder GPUs the current heuristic can fall back to Legacy, and this still
    verifies identical output.
    """
    image_batch = [
        os.path.join(img_dir_path, HW_COMPARE_SEED_IMAGES[i % len(HW_COMPARE_SEED_IMAGES)])
        for i in range(HW_TAIL_BATCH_SIZE)
    ]
    params = nvimgcodec.DecodeParams(apply_exif_orientation=True)

    try:
        batched_decoder = _make_hw_decoder("batched", cpp_threads=cpp_threads)
        batch_single_cpp_decoder = _make_hw_decoder("batch_single_cpp_threads", cpp_threads=cpp_threads)
    except RuntimeError as exc:
        if _is_hw_decoder_unavailable(exc):
            t.skip(f"nvJPEG HW decoder not available: {exc}")
        raise

    batched_images = _decode_batch_to_host(batched_decoder, image_batch, input_mode, params)
    batch_single_cpp_images = _decode_batch_to_host(batch_single_cpp_decoder, image_batch, input_mode, params)
    _assert_same_outputs("batched", batched_images, "batch_single_cpp_threads", batch_single_cpp_images, image_batch)


@t.mark.parametrize("input_mode", ["path", "python"])
def test_force_legacy_batch_option(input_mode: str):
    """Verify the customer-facing ``nvjpeg_hw_decoder:force_legacy_batch=1``
    option forces the legacy nvjpegDecodeBatched* path.

    Constructs three decoders:
      ref              -> max_num_cpu_threads=1, legacy path by configuration.
      forced_legacy    -> max_num_cpu_threads=4 + force_legacy_batch=1; this
                          forces the heuristic back to Legacy.
      batch_single_cpp -> max_num_cpu_threads=4 with no force option; the
                          mixed-CSS batch selects BatchSingle on multi-engine GPUs.

    Decoded pixel outputs must be bit-identical across all three.
    """
    assert HW_COMPARE_BATCH_SIZE % len(HW_COMPARE_SEED_IMAGES) == 0
    repeats_per_batch = HW_COMPARE_BATCH_SIZE // len(HW_COMPARE_SEED_IMAGES)
    image_batch = [
        os.path.join(img_dir_path, rel_path) for rel_path in HW_COMPARE_SEED_IMAGES
    ] * repeats_per_batch
    params = nvimgcodec.DecodeParams(apply_exif_orientation=True)

    try:
        ref_decoder = _make_hw_decoder("batched", cpp_threads=1)
        forced_legacy_decoder = nvimgcodec.Decoder(
            backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY)],
            options="nvjpeg_hw_decoder:force_legacy_batch=1",
            max_num_cpu_threads=4,
        )
        batch_single_cpp_decoder = _make_hw_decoder("batch_single_cpp_threads", cpp_threads=4)
    except RuntimeError as exc:
        if _is_hw_decoder_unavailable(exc):
            t.skip(f"nvJPEG HW decoder not available: {exc}")
        raise

    ref_images = _decode_batch_to_host(ref_decoder, image_batch, input_mode, params)
    forced_legacy_images = _decode_batch_to_host(forced_legacy_decoder, image_batch, input_mode, params)
    batch_single_cpp_images = _decode_batch_to_host(batch_single_cpp_decoder, image_batch, input_mode, params)
    _assert_same_outputs("batched", ref_images, "force_legacy_batch=1", forced_legacy_images, image_batch)
    _assert_same_outputs("batched", ref_images, "batch_single_cpp_threads", batch_single_cpp_images, image_batch)


@t.mark.parametrize("input_mode", ["path", "python"])
def test_force_batch_single_option(input_mode: str):
    """Verify ``nvjpeg_hw_decoder:force_batch_single=1`` decodes identically to
    the legacy batched path.

    This makes BatchSingle coverage explicit instead of relying only on the
    mixed-CSS heuristic, which can intentionally choose Legacy on some HW.
    """
    assert HW_COMPARE_BATCH_SIZE % len(HW_COMPARE_SEED_IMAGES) == 0
    repeats_per_batch = HW_COMPARE_BATCH_SIZE // len(HW_COMPARE_SEED_IMAGES)
    image_batch = [
        os.path.join(img_dir_path, rel_path) for rel_path in HW_COMPARE_SEED_IMAGES
    ] * repeats_per_batch
    params = nvimgcodec.DecodeParams(apply_exif_orientation=True)

    try:
        ref_decoder = _make_hw_decoder("batched", cpp_threads=1)
        forced_batch_single_decoder = _make_hw_decoder("force_batch_single", cpp_threads=4)
    except RuntimeError as exc:
        if _is_hw_decoder_unavailable(exc):
            t.skip(f"nvJPEG HW decoder not available: {exc}")
        raise

    ref_images = _decode_batch_to_host(ref_decoder, image_batch, input_mode, params)
    forced_batch_single_images = _decode_batch_to_host(forced_batch_single_decoder, image_batch, input_mode, params)
    _assert_same_outputs("batched", ref_images, "force_batch_single=1", forced_batch_single_images, image_batch)
