@echo on
REM Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM Licensed under the Apache License, Version 2.0 (the "License");

setlocal enabledelayedexpansion

mkdir build

cd build

if errorlevel 1 exit 1

set "NVIMGCODEC_DIR="
set /a NVIMGCODEC_DIR_MATCHES=0
set "NVIMGCODEC_DIR_MATCH_LIST="
for /d %%D in ("%LIBRARY_LIB%\libnvimgcodec\*") do (
    if exist "%%~fD\cmake\nvimgcodec\nvimgcodecConfig.cmake" (
        set /a NVIMGCODEC_DIR_MATCHES+=1
        set "NVIMGCODEC_DIR=%%~fD\cmake\nvimgcodec"
        set "NVIMGCODEC_DIR_MATCH_LIST=!NVIMGCODEC_DIR_MATCH_LIST! %%~fD\cmake\nvimgcodec"
    )
)
if not "!NVIMGCODEC_DIR_MATCHES!"=="1" (
    echo Expected exactly one nvimgcodecConfig.cmake under "%LIBRARY_LIB%\libnvimgcodec", found !NVIMGCODEC_DIR_MATCHES! >&2
    echo Matches:!NVIMGCODEC_DIR_MATCH_LIST! >&2
    exit 1
)

:: To speed-up debugging, build for one arch
:: set CUDAARCHS=50

set NVIMG_BUILD_ARGS= ^
    -DBUILD_DOCS:BOOL=OFF ^
    -DBUILD_SAMPLES:BOOL=OFF ^
    -DBUILD_TEST:BOOL=OFF ^
    -DCUDA_TARGET_ARCHS=%CUDAARCHS%

set NVIMG_LIBRARY_ARGS= ^
    -DBUILD_LIBRARY:BOOL=OFF ^
    -DBUILD_SHARED_LIBS:BOOL=ON ^
    -DBUILD_STATIC_LIBS:BOOL=OFF ^
    -DWITH_DYNAMIC_LINK:BOOL=OFF ^
    -DWITH_SHARED_CUDA_LIBS:BOOL=ON

set NVIMG_EXT_ARGS= ^
    -DBUILD_EXTENSIONS:BOOL=OFF

set NVIMG_PYTHON_ARGS= ^
    -DBUILD_PYTHON:BOOL=ON ^
    -DPYTHON_VERSIONS=%PY_VER% ^
    -DBUILD_WHEEL:BOOL=OFF ^
    -DNVIMGCODEC_COPY_LIBS_TO_PYTHON_DIR:BOOL=OFF ^
    -DNVIMGCODEC_BUILD_PYBIND11:BOOL=OFF ^
    -DNVIMGCODEC_BUILD_DLPACK:BOOL=OFF

cmake %CMAKE_ARGS% -GNinja -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
    -Dnvimgcodec_DIR="!NVIMGCODEC_DIR!" ^
    %NVIMG_BUILD_ARGS% %NVIMG_LIBRARY_ARGS% %NVIMG_EXT_ARGS% ^
    %NVIMG_PYTHON_ARGS% %SRC_DIR%

if errorlevel 1 exit 1

cmake --build .

if errorlevel 1 exit 1

cmake --install .

if errorlevel 1 exit 1

@REM copy in CMakeLists doesn't work for unknown reason
move .\python\*.pyd .\python\nvidia\nvimgcodec\

if errorlevel 1 exit 1

%PYTHON% -m pip install --no-deps --no-build-isolation -v %SRC_DIR%/build/python

if errorlevel 1 exit 1

endlocal
