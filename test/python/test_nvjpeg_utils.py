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

from types import SimpleNamespace

import utils


def test_linux_only_reuses_an_already_loaded_library(monkeypatch):
    """Use the CUDA-major SONAME first and require RTLD_NOLOAD on every lookup."""
    calls = []
    loaded_library = object()

    def no_load_cdll(library_name, mode):
        calls.append((library_name, mode))
        if library_name == "libnvjpeg.so.13":
            raise OSError("versioned SONAME is not loaded")
        return loaded_library

    monkeypatch.setattr(utils.os, "RTLD_LAZY", 0x1, raising=False)
    monkeypatch.setattr(utils.os, "RTLD_NOLOAD", 0x4, raising=False)
    monkeypatch.setattr(utils.c, "CDLL", no_load_cdll)

    libraries = list(utils._loaded_nvjpeg_libraries("13", "Linux"))

    assert libraries == [loaded_library]
    assert calls == [
        ("libnvjpeg.so.13", 0x5),
        ("libnvjpeg.so", 0x5),
    ]


def test_windows_wraps_the_borrowed_module_handle(monkeypatch):
    """Get the loaded CUDA-major DLL and pass its handle directly to ctypes."""
    module_lookups = []
    cdll_calls = []
    loaded_library = object()

    def get_module_handle(library_name):
        module_lookups.append(library_name)
        return 0x1234 if library_name == "nvjpeg64_13.dll" else None

    def wrap_handle(library_name, handle):
        cdll_calls.append((library_name, handle))
        return loaded_library

    kernel32 = SimpleNamespace(GetModuleHandleW=get_module_handle)
    monkeypatch.setattr(utils.c, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    monkeypatch.setattr(utils.c, "CDLL", wrap_handle)

    libraries = list(utils._loaded_nvjpeg_libraries("13", "Windows"))

    assert libraries == [loaded_library]
    assert module_lookups == ["nvjpeg64_13.dll", "nvjpeg.dll"]
    assert cdll_calls == [("nvjpeg64_13.dll", 0x1234)]


def test_get_nvjpeg_ver_skips_library_with_failed_property_queries(monkeypatch):
    """Skip a loaded library that cannot report its full version."""
    def library(status, version):
        def get_property(index, value):
            value._obj.value = version[index]
            return status

        return SimpleNamespace(nvjpegGetProperty=get_property)

    libraries = [
        library(1, (12, 9, 0)),
        library(0, (13, 2, 1)),
    ]
    monkeypatch.setattr(utils, "_loaded_nvjpeg_libraries", lambda *_: libraries)

    assert utils.get_nvjpeg_ver() == (13, 2, 1)
