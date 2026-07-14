#!/bin/bash -ex

export VERSION=${VERSION:-release_0.9.0}  # Update version when changing anything in the Dockerfiles
export TEGRA_VERSION=${TEGRA_VERSION:-26} # Update version when changing anything in the Dockerfile.tegra-aarch64-linux.builder
export CONDA_TEST_VERSION=${CONDA_TEST_VERSION:-2} # Update version when changing anything in the Dockerfile.conda-test

SCRIPT_DIR=$(dirname $0)
source ${SCRIPT_DIR}/config-docker.sh || source ${SCRIPT_DIR}/default-config-docker.sh


docker buildx create --name nvimagecodec_builder || echo "nvimagecodec_build already created"
docker buildx use nvimagecodec_builder
docker buildx inspect --bootstrap

####### BASE IMAGES #######

# Manylinux_2_28 with GCC 14
export MANYLINUX_GCC14="${REGISTRY_PREFIX}manylinux_2_28_${ARCH}.gcc14"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${MANYLINUX_GCC14} \
    -t ${MANYLINUX_GCC14} -t ${MANYLINUX_GCC14}:v${VERSION} \
    -f docker/Dockerfile.gcc14 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux_2_28_${ARCH}:${MANYLINUX_IMAGE_TAG}" \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 12.9.0
export CUDA_129="${REGISTRY_PREFIX}cuda12.9-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${CUDA_129} \
    -t ${CUDA_129} -t ${CUDA_129}:v${VERSION} \
    -f docker/Dockerfile.cuda129.${ARCH}.deps \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 13.3.0
export CUDA_133="${REGISTRY_PREFIX}cuda13.3-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from "type=registry,ref=${CUDA_133}" \
    -t "${CUDA_133}" -t "${CUDA_133}:v${VERSION}" \
    -f "docker/Dockerfile.cuda133.${ARCH}.deps" \
    --platform "${PLATFORM}" \
    --push \
    .

# GCC 14
export DEPS_GCC14="${REGISTRY_PREFIX}nvimgcodec_deps-${ARCH}-manylinux_2_28-gcc14"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${DEPS_GCC14} \
    -t ${DEPS_GCC14} -t ${DEPS_GCC14}:v${VERSION} \
    -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC14}:v${VERSION}" \
    --build-arg "ARCH=${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    .

if [ "$ARCH" == "x86_64" ] && [ -z "$FREE_THREADED_PYTHON_IMAGE" ]; then
    export CUDA_VERSION=12.5.0
    export FREE_THREADED_PYTHON_IMAGE="${REGISTRY_PREFIX}free-threaded-python-cuda-12.5-x86_64-py313t"
    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref=${FREE_THREADED_PYTHON_IMAGE} \
        --build-arg CUDA_VERSION=${CUDA_VERSION} \
        -t ${FREE_THREADED_PYTHON_IMAGE} \
        -t ${FREE_THREADED_PYTHON_IMAGE}:v${VERSION} \
        --platform "linux/amd64" \
        --push \
        https://github.com/NVIDIA/free-threaded-python.git#main:docker
fi

####### BUILDER IMAGES #######

# GCC 14, CUDA 12.9
export BUILDER_CUDA_129="${REGISTRY_PREFIX}builder-cuda-12.9-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${BUILDER_CUDA_129} \
    -t ${BUILDER_CUDA_129} -t ${BUILDER_CUDA_129}:v${VERSION} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC14}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_129}:v${VERSION}" \
    --platform ${PLATFORM} \
    --push \
    .

# GCC 14, CUDA 13.3
export BUILDER_CUDA_133="${REGISTRY_PREFIX}builder-cuda-13.3-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from "type=registry,ref=${BUILDER_CUDA_133}" \
    -t "${BUILDER_CUDA_133}" -t "${BUILDER_CUDA_133}:v${VERSION}" \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC14}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_133}:v${VERSION}" \
    --platform ${PLATFORM} \
    --push \
    .


# Cross-compiling host=x86_64 target=L4T
if [ "$ARCH" == "x86_64" ]; then
    export BUILDER_CUDA_TEGRA_129="${REGISTRY_PREFIX}builder-cuda-12.9-cross-l4t-aarch64-linux"
    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref=${BUILDER_CUDA_TEGRA_129} \
        -t ${BUILDER_CUDA_TEGRA_129} -t ${BUILDER_CUDA_TEGRA_129}:tegra_v${TEGRA_VERSION} \
        -f docker/Dockerfile.tegra-aarch64-linux.builder \
        --build-arg "CUDA_CROSS_VERSION=12-9" \
        --build-arg "CUDA_CROSS_VERSION_DOT=12.9" \
        --build-arg "CUDA_TARGET_ARCHS=72;87" \
        --platform ${PLATFORM} \
        --push \
        .
fi

####### TEST IMAGES #######

# Note: we are using devel image because we need CTK to be available for cupy to be able to JIT compile kernels.

# CUDA 12.9
export RUNNER_CUDA_129="${REGISTRY_PREFIX}runner-cuda-12.9-${ARCH}-${PYVER}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${RUNNER_CUDA_129} \
    -t ${RUNNER_CUDA_129} -t ${RUNNER_CUDA_129}:v${VERSION} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:12.9.0-devel-ubuntu24.04" \
    --build-arg "VER_CUDA=12.9.0" \
    --build-arg "VER_UBUNTU=24.04" \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 12.5, Python 3.13t (free-threaded)
if [ "$ARCH" == "x86_64" ]; then
    export RUNNER_CUDA_125_PY313T="${REGISTRY_PREFIX}runner-cuda-12.5-x86_64-py313t"
    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref=${RUNNER_CUDA_125_PY313T} \
        -t ${RUNNER_CUDA_125_PY313T} -t ${RUNNER_CUDA_125_PY313T}:v${VERSION} \
        --platform "linux/amd64" \
        --push \
        -<<EOF
FROM $FREE_THREADED_PYTHON_IMAGE
RUN python -m pip uninstall -y nvidia-nvimgcodec-cu12 || true
EOF

fi

# CUDA 13.3
export RUNNER_CUDA_133="${REGISTRY_PREFIX}runner-cuda-13.3-${ARCH}-${PYVER}"
docker buildx build \
    --cache-to type=inline \
    --cache-from "type=registry,ref=${RUNNER_CUDA_133}" \
    -t "${RUNNER_CUDA_133}" -t "${RUNNER_CUDA_133}:v${VERSION}" \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:13.3.0-devel-ubuntu24.04" \
    --build-arg "VER_CUDA=13.3.0" \
    --build-arg "VER_UBUNTU=24.04" \
    --platform ${PLATFORM} \
    --push \
    .

####### CONDA TEST IMAGES #######

if [ "$ARCH" == "x86_64" ]; then
    # Dockerfile.conda-test has no COPY/ADD, so the build context is irrelevant.
    # Pipe the Dockerfile via stdin (context arg `-`) to avoid uploading the
    # entire repo to the Docker daemon on every build.

    # CUDA 12.9
    export CONDA_TEST_CUDA_129="${REGISTRY_PREFIX}conda-test-cuda-12.9-x86_64"
    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref=${CONDA_TEST_CUDA_129} \
        -t ${CONDA_TEST_CUDA_129} -t ${CONDA_TEST_CUDA_129}:conda_test_v${CONDA_TEST_VERSION} \
        --build-arg "CUDA_VERSION=12.9.0" \
        --platform "linux/amd64" \
        --push \
        - < docker/Dockerfile.conda-test

    # CUDA 13.3
    export CONDA_TEST_CUDA_133="${REGISTRY_PREFIX}conda-test-cuda-13.3-x86_64"
    docker buildx build \
        --cache-to type=inline \
        --cache-from "type=registry,ref=${CONDA_TEST_CUDA_133}" \
        -t "${CONDA_TEST_CUDA_133}" -t "${CONDA_TEST_CUDA_133}:conda_test_v${CONDA_TEST_VERSION}" \
        --build-arg "CUDA_VERSION=13.3.0" \
        --platform "linux/amd64" \
        --push \
        - < docker/Dockerfile.conda-test
fi
