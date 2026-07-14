/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdlib>
#include <iostream>
#include <string>
#include <optional>

#include <pybind11/stl_bind.h>

#include <ilogger.h>
#include <log.h>

#include <nvimgcodec.h>
#include "image.h"
#include "module.h"

namespace nvimgcodec {

uint32_t verbosity2severity(int verbose)
{
    uint32_t result = 0;
    if (verbose >= 1)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR;
    if (verbose >= 2)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING;
    if (verbose >= 3)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO;
    if (verbose >= 4)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG;
    if (verbose >= 5)
        result |= NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE;

    return result;
}

Module::Module()
    : dbg_messenger_handle_(nullptr)
{
    int verbosity = 2;
    std::string verbosity_warning;
    char* v = std::getenv("PYNVIMGCODEC_VERBOSITY");
    try {
        if (v) {
            verbosity = std::stoi(v);
        }
    } catch (std::invalid_argument const& ex) {
        verbosity_warning = "PYNVIMGCODEC_VERBOSITY has wrong value";
    } catch (std::out_of_range const& ex) {
        verbosity_warning = "PYNVIMGCODEC_VERBOSITY has out of range value";
    }

    if (verbosity > 0) {
        dbg_messenger_ = std::make_unique<DefaultDebugMessenger>(verbosity2severity(verbosity), NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL);
        logger_ = std::make_unique<Logger>("pynvimgcodec", dbg_messenger_.get());

        if (!verbosity_warning.empty()) {
            NVIMGCODEC_LOG_WARNING(logger_.get(), verbosity_warning);
        }
    } else {
        logger_ = std::make_unique<Logger>("pynvimgcodec");
    }

    nvimgcodecInstanceCreateInfo_t instance_create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.load_extension_modules = 1;
    instance_create_info.create_debug_messenger = verbosity > 0 ? 1 : 0;
    instance_create_info.debug_messenger_desc = verbosity > 0 ? dbg_messenger_->getDesc() : nullptr;

    nvimgcodecInstanceCreate(&instance_, &instance_create_info);

}

Module ::~Module()
{
    nvimgcodecInstanceDestroy(instance_);
}

void Module::exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger)
{
    m.def(
         "as_image",
         [instance, logger](py::handle source, intptr_t cuda_stream,
                           std::optional<nvimgcodecSampleFormat_t> sample_format,
                           std::optional<nvimgcodecColorSpec_t> color_spec,
                           std::optional<int> precision) -> Image {
             return Image(instance, logger, source.ptr(), cuda_stream, sample_format, color_spec, precision);
         },
         R"pbdoc(
        Wraps an external buffer as an Image and ties the buffer lifetime to the Image.

        The buffer must be C-style contiguous, but rows may have additional padding.

        Args:
            source: Input DLPack tensor encapsulated in a PyCapsule, or any object
                    exposing ``__cuda_array_interface__``, ``__array_interface__``
                    or ``__dlpack__``/``__dlpack_device__``.

            cuda_stream: Optional ``cudaStream_t`` as a Python integer, used for any
                         synchronization the created Image needs.

            sample_format: (keyword-only) ``nvimgcodec.SampleFormat`` selecting the
                           output layout:

                           * ``I_*`` (concrete interleaved) -> HWC; the last axis
                             must have at least the format's channel arity (2-D
                             ``(H, W)`` only for ``I_Y``).
                           * ``P_*`` (concrete planar)      -> CHW; the leading
                             axis must have at least the format's channel arity
                             (2-D ``(H, W)`` only for ``P_Y``).
                           * ``I_UNCHANGED``                -> HWC, no arity check.
                           * ``P_UNCHANGED``                -> CHW, no arity check.
                           * ``UNKNOWN``                    -> raises ``ValueError``.

                           Extra channels beyond the format's arity are kept; too
                           few raise ``ValueError``.

                           When omitted, the value is inferred from the channel
                           count of the array using the HWC defaults:

                           * 1 channel  -> ``I_Y``
                           * 2 channels -> ``I_YA``
                           * 3 channels -> ``I_RGB``
                           * 4 channels -> ``I_RGBA``
                           * 5+ channels -> ``UNKNOWN``

            color_spec: (keyword-only) ``nvimgcodec.ColorSpec`` override. When omitted
                        the value is inferred from the channel count:

                        * 1 / 2 channels -> ``GRAY``
                        * 3 / 4 channels -> ``SRGB``
                        * 5+ channels   -> ``UNKNOWN``

            precision: (keyword-only) Optional integer giving the number of significant
                       bits per sample. Use this to describe lower-precision data
                       stored in a wider container, e.g. a 12-bit image held in a
                       uint16 buffer (``precision=12``). Accepted values are 0 (or
                       ``None``, both meaning "use the full bitdepth of the sample
                       data type") and any integer in ``1..bitdepth(dtype)``. For
                       floating-point dtypes only 0/None or the dtype's full bitdepth
                       (e.g. 32 for float32) is accepted; sub-bitdepth values are an
                       integer-image concept and have no meaningful interpretation
                       for floats.

        Returns:
            nvimgcodec.Image

        )pbdoc",
         "source"_a, "cuda_stream"_a = 0, py::kw_only(), "sample_format"_a = py::none(), "color_spec"_a = py::none(),
         "precision"_a = py::none(), py::keep_alive<0, 1>())
        .def(
            "as_images",
            [instance, logger](const std::vector<py::handle>& sources, intptr_t cuda_stream,
                              std::optional<nvimgcodecSampleFormat_t> sample_format,
                              std::optional<nvimgcodecColorSpec_t> color_spec,
                              std::optional<int> precision) -> std::vector<py::object> {
                std::vector<py::object> py_images;
                py_images.reserve(sources.size());
                for (auto& source : sources) {
                    Image img(instance, logger, source.ptr(), cuda_stream, sample_format, color_spec, precision);
                    py::object py_img = py::cast(img);
                    py_images.push_back(py_img);
                    py::detail::keep_alive_impl(py_img, source);
                }
                return py_images;
            },
            R"pbdoc(
            Wrap a list of external buffers as Images and tie each buffer's lifetime
            to the corresponding Image. Equivalent to calling :func:`as_image` on
            each element of ``sources`` with the same ``sample_format``, ``color_spec``
            and ``precision`` overrides.

            Layout rules and sample_format / color_spec / precision inference are
            identical to :func:`as_image`. The same overrides are applied uniformly
            to every input - the elements of ``sources`` may otherwise differ in
            shape, dtype and channel count.

            Args:
                sources: List of input DLPack tensors (PyCapsules) or objects exposing
                         ``__cuda_array_interface__``, ``__array_interface__`` or
                         ``__dlpack__``/``__dlpack_device__``.

                cuda_stream: Optional ``cudaStream_t`` as a Python integer, used for
                             any synchronization the created Images need.

                sample_format: (keyword-only) ``nvimgcodec.SampleFormat`` override
                               applied to every Image, selecting the output layout:

                               * ``I_*`` (concrete interleaved) -> HWC; the last axis
                                 must have at least the format's channel arity (2-D
                                 ``(H, W)`` only for ``I_Y``).
                               * ``P_*`` (concrete planar)      -> CHW; the leading
                                 axis must have at least the format's channel arity
                                 (2-D ``(H, W)`` only for ``P_Y``).
                               * ``I_UNCHANGED``                -> HWC, no arity check.
                               * ``P_UNCHANGED``                -> CHW, no arity check.
                               * ``UNKNOWN``                    -> raises ``ValueError``.

                               Extra channels beyond the format's arity are kept; too
                               few raise ``ValueError``.

                               When omitted, each Image's value is inferred from its
                               own channel count using the HWC defaults:

                               * 1 channel  -> ``I_Y``
                               * 2 channels -> ``I_YA``
                               * 3 channels -> ``I_RGB``
                               * 4 channels -> ``I_RGBA``
                               * 5+ channels -> ``UNKNOWN``

                color_spec: (keyword-only) ``nvimgcodec.ColorSpec`` override applied
                            to every Image. When omitted, each Image's value is
                            inferred from its own channel count:

                            * 1 / 2 channels -> ``GRAY``
                            * 3 / 4 channels -> ``SRGB``
                            * 5+ channels   -> ``UNKNOWN``

                precision: (keyword-only) Optional integer giving the number of
                           significant bits per sample, applied uniformly to every
                           Image. Use this to describe lower-precision data stored in
                           a wider container, e.g. a 12-bit image held in a uint16
                           buffer (``precision=12``). Accepted values are 0 (or
                           ``None``, both meaning "use the full bitdepth of the sample
                           data type") and any integer in ``1..bitdepth(dtype)``. For
                           floating-point dtypes only 0/None or the dtype's full
                           bitdepth (e.g. 32 for float32) is accepted; sub-bitdepth
                           values are an integer-image concept and have no meaningful
                           interpretation for floats. The value is validated against
                           each source's dtype.

            Returns:
                List of nvimgcodec.Image objects.
            )pbdoc",
            "sources"_a, "cuda_stream"_a = 0, py::kw_only(), "sample_format"_a = py::none(), "color_spec"_a = py::none(),
            "precision"_a = py::none())
        .def(
            "from_dlpack",
            [instance, logger](py::handle source, intptr_t cuda_stream,
                               std::optional<nvimgcodecSampleFormat_t> sample_format,
                               std::optional<nvimgcodecColorSpec_t> color_spec,
                               std::optional<int> precision) -> Image {
                return Image(instance, logger, source.ptr(), cuda_stream, sample_format, color_spec, precision);
            },
            R"pbdoc(
            Zero-copy conversion from a DLPack tensor to an Image.

            The DLPack source must be a 3-dimensional, CUDA-accessible tensor
            (device or CUDA-host memory). Layout selection and sample_format /
            color_spec inference are identical to :func:`as_image`; see that
            function for the full rules.

            Args:
                source: Input DLPack tensor encapsulated in a PyCapsule, or any object
                        exposing ``__dlpack__`` and ``__dlpack_device__`` methods.

                cuda_stream: Optional ``cudaStream_t`` as a Python integer, used for any
                             synchronization the created Image needs.

                sample_format: (keyword-only) ``nvimgcodec.SampleFormat`` selecting the
                               output layout:

                               * ``I_*`` (concrete interleaved) -> HWC; the last axis
                                 must have at least the format's channel arity.
                               * ``P_*`` (concrete planar)      -> CHW; the leading
                                 axis must have at least the format's channel arity.
                               * ``I_UNCHANGED``                -> HWC, no arity check.
                               * ``P_UNCHANGED``                -> CHW, no arity check.
                               * ``UNKNOWN``                    -> raises ``ValueError``.

                               Extra channels beyond the format's arity are kept; too
                               few raise ``ValueError``.

                               When omitted, the value is inferred from the channel
                               count of the tensor using the HWC defaults:

                               * 1 channel  -> ``I_Y``
                               * 2 channels -> ``I_YA``
                               * 3 channels -> ``I_RGB``
                               * 4 channels -> ``I_RGBA``
                               * 5+ channels -> ``UNKNOWN``

                color_spec: (keyword-only) ``nvimgcodec.ColorSpec`` override. When omitted
                            the value is inferred from the channel count:

                            * 1 / 2 channels -> ``GRAY``
                            * 3 / 4 channels -> ``SRGB``
                            * 5+ channels   -> ``UNKNOWN``

                precision: (keyword-only) Optional integer giving the number of significant
                           bits per sample. Use this to describe lower-precision data
                           stored in a wider container, e.g. a 12-bit image held in a
                           uint16 buffer (``precision=12``). Accepted values are 0 (or
                           ``None``, both meaning "use the full bitdepth of the sample
                           data type") and any integer in ``1..bitdepth(dtype)``. See
                           :func:`as_image` for details.

            Returns:
                nvimgcodec.Image

            )pbdoc",
            "source"_a, "cuda_stream"_a = 0, py::kw_only(), "sample_format"_a = py::none(), "color_spec"_a = py::none(),
            "precision"_a = py::none(), py::keep_alive<0, 1>());
}

} // namespace nvimgcodec
