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

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest as t

# Add the test/python directory to sys.path to allow imports of local test helpers.
test_python_dir = Path(__file__).parent
if str(test_python_dir) not in sys.path:
    sys.path.insert(0, str(test_python_dir))


@t.fixture(scope="session", autouse=True)
def _nvjpeg_session_keepalive():
    """Hold one HW-capable Decoder for the duration of the pytest session.

    The nvjpeg HW SharedContext (CUDA streams, nvJPEG handle, worker pool)
    is created on first Decoder construction and destroyed when the last
    Decoder on that GPU is released.  Without this fixture, tests that
    create and drop Decoders pay the full construction cost every time —
    measured ~3x slower test suite on marie RTX 6000 Ada with HW backend.

    The fixture creates a dummy Decoder and one warmup decode at session
    start, holding the SharedContext alive via shared_ptr through the
    entire pytest run.  Individual tests' Decoders share that context
    (cheap reuse via the weak_ptr registry).  At session teardown the
    fixture drops the warmup output then the decoder while pytest is
    still in control — before any global static / interpreter shutdown
    order issues.

    Wrapped in try/except so machines without a working nvJPEG HW
    backend (e.g. workstation Ampere parts) silently skip the warmup
    and the rest of the test session runs unaffected.
    """
    decoder = None
    decoded = None
    try:
        from nvidia import nvimgcodec
        from utils import img_dir_path
        decoder = nvimgcodec.Decoder()
        sample_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_420.jpg")
        if os.path.exists(sample_path):
            sample_bytes = np.fromfile(sample_path, dtype=np.uint8)
            decoded = decoder.decode(sample_bytes)
    except Exception:
        decoder = None
        decoded = None

    try:
        yield
    finally:
        # Tear down warmup output first, then the decoder, while pytest
        # is still in control rather than relying on interpreter shutdown
        # order with a possibly-already-destroyed CUDA context.
        decoded = None
        decoder = None
        gc.collect()
