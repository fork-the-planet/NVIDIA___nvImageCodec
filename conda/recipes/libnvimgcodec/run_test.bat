setlocal EnableDelayedExpansion
if "%PKG_NAME%"=="libnvimgcodec-dev" (
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
        exit /b 1
    )
    cmake %CMAKE_ARGS% -Dnvimgcodec_DIR="!NVIMGCODEC_DIR!" -GNinja %NVIMG_CTK_ARGS% test
    if errorlevel 1 exit /b 1
    cmake --build .
    if errorlevel 1 exit /b 1
)
endlocal
