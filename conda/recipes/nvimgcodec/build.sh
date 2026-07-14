#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# To speed-up debugging, build for one arch
# export CUDAARCHS="50"

# nvimagecodec has a custom wheel building logic instead of using standard tools, so we have
# to set these environment variables in order for the wheel metadata to be correct when
# cross-compiling
if [[ "$target_platform" == "linux-64" ]]; then
        export ARCH="x86_64"
        export WHL_PLATFORM_NAME="manylinux_${c_stdlib_version//./_}_${ARCH}"
elif [[ "$target_platform" == "linux-aarch64" ]]; then
        export ARCH="aarch64"
        export WHL_PLATFORM_NAME="manylinux_${c_stdlib_version//./_}_${ARCH}"
else
    echo "$target_platform is an unknown target_platform!"
    exit 1
fi

mkdir build
cd build

shopt -s nullglob
nvimgcodec_cmake_candidates=("${PREFIX}"/lib{,64}/libnvimgcodec/*/cmake/nvimgcodec)
shopt -u nullglob

nvimgcodec_cmake_dirs=()
for nvimgcodec_cmake_candidate in "${nvimgcodec_cmake_candidates[@]}"; do
    if [[ -f "${nvimgcodec_cmake_candidate}/nvimgcodecConfig.cmake" ]]; then
        nvimgcodec_cmake_dirs+=("${nvimgcodec_cmake_candidate}")
    fi
done

if [[ ${#nvimgcodec_cmake_dirs[@]} -ne 1 ]]; then
    echo "Expected exactly one nvimgcodecConfig.cmake under ${PREFIX}/lib{,64}/libnvimgcodec, found ${#nvimgcodec_cmake_dirs[@]}" >&2
    printf '  %s\n' "${nvimgcodec_cmake_dirs[@]}" >&2
    exit 1
fi

nvimg_build_args=(
    -DBUILD_DOCS:BOOL=OFF
    -DBUILD_SAMPLES:BOOL=OFF
    -DBUILD_TEST:BOOL=OFF
    -DCUDA_TARGET_ARCHS=${CUDAARCHS}
# Library args
    -DBUILD_LIBRARY:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DBUILD_STATIC_LIBS:BOOL=OFF
# "DYNAMIC_LINK" means using dlopen, but we want to link to shared libraries.
    -DWITH_DYNAMIC_LINK:BOOL=OFF
    -DWITH_SHARED_CUDA_LIBS:BOOL=ON
# Stay inside the conda env (see libnvimgcodec recipe for rationale).
    -DCMAKE_IGNORE_PREFIX_PATH=/usr/local
    -Dnvimgcodec_DIR="${nvimgcodec_cmake_dirs[0]}"
# Extension args
    -DBUILD_EXTENSIONS:BOOL=OFF
# Python args
    -DBUILD_PYTHON:BOOL=ON
    -DPYTHON_VERSIONS="${PY_VER}"
    -DBUILD_WHEEL:BOOL=OFF
    -DNVIMGCODEC_COPY_LIBS_TO_PYTHON_DIR:BOOL=OFF
    -DNVIMGCODEC_BUILD_PYBIND11:BOOL=OFF
    -DNVIMGCODEC_BUILD_DLPACK:BOOL=OFF
    -DARCH="${ARCH}"
    -DNVIMGCODEC_WHL_PLATFORM_NAME="${WHL_PLATFORM_NAME}"
)

cmake ${CMAKE_ARGS} -GNinja "${nvimg_build_args[@]}" ${SRC_DIR}

cmake --build . --verbose

cmake --install . --strip

# When cross-compiling, the python modules are named incorrectly, so we have to
# fix the name.
if [[ "$target_platform" == "linux-aarch64" ]]; then
  for file in "${SRC_DIR}"/build/python/nvidia/nvimgcodec/*cpython-*-x86_64-linux-gnu.so; do
    newname="${file/x86_64/aarch64}"
    mv "$file" "$newname"
    echo "Renamed: $file → $newname"
  done
fi

$PYTHON -m pip install --no-deps --no-build-isolation -v $SRC_DIR/build/python

# Just double checking that binaries target correct arch
file ${SP_DIR}/nvidia/nvimgcodec/*.so
ldd ${SP_DIR}/nvidia/nvimgcodec/*.so | tee nvimgcodec_impl.ldd
! grep -q "libnvimgcodec.*not found" nvimgcodec_impl.ldd
