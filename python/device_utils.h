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

#pragma once

#include <cuda_runtime.h>
#include <nvimgcodec.h>

#include "error_handling.h"

// Resolves ``NVIMGCODEC_DEVICE_CURRENT`` (-1) to the actual current CUDA
// device ordinal. Other values (valid device ids or
// ``NVIMGCODEC_DEVICE_CPU_ONLY``) are returned unchanged.
inline int resolve_device_id(int device_id)
{
    if (device_id == NVIMGCODEC_DEVICE_CURRENT) {
        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));
        return dev;
    }
    return device_id;
}
