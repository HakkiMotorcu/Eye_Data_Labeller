@echo off
REM ==================================================================
REM  Eye Data Labeller -- one-shot installer for Windows.
REM
REM  Steps:
REM    1. Download Miniforge into %USERPROFILE%\miniforge3 if missing.
REM    2. Create / update the `eye-labeller` conda env from
REM       environment.yml (CPU PyTorch baseline -- same on every OS).
REM    3. Detect NVIDIA GPU via nvidia-smi.exe; if present, swap to
REM       the CUDA-enabled pip wheel of PyTorch.
REM    4. Drop a launcher at <Desktop>\EyeDataLabeller.bat (real
REM       Desktop -- OneDrive-redirected folders included).
REM
REM  CUDA WHEEL VERSION:
REM    Default cu124 (CUDA 12.4) works with NVIDIA driver >=550.
REM    For older drivers, edit CUDA_INDEX below to cu121 (driver
REM    >=525). Open Command Prompt and run `nvidia-smi` to see your
REM    driver version on the top line of the output.
REM ==================================================================

REM Plain setlocal, NOT enabledelayedexpansion: nothing here uses
REM !var! expansion, and delayed expansion silently EATS '!' characters
REM from every %VAR% / %%d substitution (a Desktop path or best.pt
REM path containing '!' would be corrupted).
setlocal

set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fi"
set "MINIFORGE_DIR=%USERPROFILE%\miniforge3"
set "ENV_NAME=eye-labeller"

REM These are constants -- they MUST be set here at top level, never
REM inside a parenthesized if/else block: cmd expands %VAR% in a block
REM when the block is PARSED, so `set` + use in the same block reads
REM an empty string (this exact bug broke every fresh install).
set "MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
set "TMP_EXE=%TEMP%\miniforge-installer.exe"

REM CUDA wheel index. Change to cu121 for older drivers (see header).
set "CUDA_INDEX=https://download.pytorch.org/whl/cu124"

REM Resolve the REAL Desktop folder. On OneDrive-managed machines
REM (most university setups) it is NOT %USERPROFILE%\Desktop.
set "DESKTOP_DIR="
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP_DIR=%%d"
if not defined DESKTOP_DIR set "DESKTOP_DIR=%USERPROFILE%\Desktop"
set "DESKTOP_LAUNCHER=%DESKTOP_DIR%\EyeDataLabeller.bat"

echo [install] PROJECT_ROOT = "%PROJECT_ROOT%"
echo [install] MINIFORGE_DIR = "%MINIFORGE_DIR%"
echo [install] Desktop = "%DESKTOP_DIR%"

REM ---- 1. Miniforge ------------------------------------------------
if exist "%MINIFORGE_DIR%\Scripts\conda.exe" (
    echo [install] Miniforge already present -- reusing.
) else (
    echo [install] Downloading Miniforge installer...
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri '%MINIFORGE_URL%' -OutFile '%TMP_EXE%'" || goto :fail
    echo [install] Installing Miniforge silently...
    start /wait "" "%TMP_EXE%" /S /AddToPath=0 /RegisterPython=0 /D=%MINIFORGE_DIR%
    del "%TMP_EXE%"
)

REM Make conda available in this shell.
call "%MINIFORGE_DIR%\Scripts\activate.bat" || goto :fail

REM ---- 2. Conda env ------------------------------------------------
if not exist "%PROJECT_ROOT%\environment.yml" (
    echo [install] FAILED: environment.yml not found at "%PROJECT_ROOT%"
    goto :fail
)

call conda env list | findstr /B /C:"%ENV_NAME% " >nul
if errorlevel 1 (
    echo [install] Creating '%ENV_NAME%' env from environment.yml  ^(5-15 min^)
    call conda env create -n "%ENV_NAME%" -f "%PROJECT_ROOT%\environment.yml" || goto :fail
) else (
    echo [install] Updating existing '%ENV_NAME%' env from environment.yml
    call conda env update -n "%ENV_NAME%" -f "%PROJECT_ROOT%\environment.yml" --prune || goto :fail
    REM environment.yml switched pyqtdarktheme -> pyqtdarktheme-fork
    REM (same qdarktheme module). --prune doesn't remove pip packages,
    REM so drop the old dist to avoid two dists owning the same files.
    call conda activate "%ENV_NAME%"
    pip uninstall -y pyqtdarktheme >nul 2>&1
    call conda deactivate
)

REM micro-sam, slim install: NOT in environment.yml because both the
REM conda package and upstream's pip metadata hard-depend on napari (a
REM second GUI stack the app never uses, and the source of the Windows
REM Qt/ICU DLL conflicts). --no-deps installs just the package; its real
REM runtime deps are pinned in the yml.
echo [install] Installing micro-sam (slim, --no-deps)
call conda activate "%ENV_NAME%"
pip install --no-deps "git+https://github.com/computational-cell-analytics/micro-sam.git@v1.7.7" || goto :fail
call conda deactivate

REM ---- 3. GPU detection + CUDA PyTorch swap -----------------------
REM env.yml ships CPU PyTorch as the baseline so every platform starts
REM the same. On Windows + NVIDIA we swap to the CUDA wheel.
REM Windows CUDA wheels bundle their CUDA libraries, so --no-deps is
REM safe here (unlike Linux) and avoids disturbing numpy / scipy.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [install] No NVIDIA GPU detected -- keeping CPU PyTorch.
    echo [install]   App will run on CPU. Slower than GPU but works.
) else (
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo [install] nvidia-smi present but failed -- driver issue?
        echo [install] Skipping CUDA swap; app will use CPU.
    ) else (
        echo [install] NVIDIA GPU detected -- swapping to CUDA PyTorch
        echo [install]   Using wheel index %CUDA_INDEX%
        call conda activate "%ENV_NAME%"
        pip install --force-reinstall --no-deps torch torchvision --index-url %CUDA_INDEX% || (
            echo [install] CUDA PyTorch install FAILED -- app will fall back to CPU.
            echo [install] Check driver/CUDA compatibility per the header notes.
        )
        call conda deactivate
    )
)

REM ---- 4. Desktop launcher ----------------------------------------
REM Written as sequential appends, NOT one (echo ...) block: a project
REM path containing ')' -- e.g. anything under a folder with
REM parentheses -- terminates a parenthesized block early and corrupts
REM the launcher.
echo [install] Writing launcher to "%DESKTOP_LAUNCHER%"
if exist "%DESKTOP_LAUNCHER%" del "%DESKTOP_LAUNCHER%"
>"%DESKTOP_LAUNCHER%"  echo @echo off
>>"%DESKTOP_LAUNCHER%" echo REM Auto-generated by deploy\install_windows.bat
>>"%DESKTOP_LAUNCHER%" echo REM Clear any Qt env vars set by the user's profile -- PyQt6
>>"%DESKTOP_LAUNCHER%" echo REM should find plugins in OUR env, not elsewhere.
>>"%DESKTOP_LAUNCHER%" echo set "QT_QPA_PLATFORM_PLUGIN_PATH="
>>"%DESKTOP_LAUNCHER%" echo set "QT_PLUGIN_PATH="
>>"%DESKTOP_LAUNCHER%" echo call "%MINIFORGE_DIR%\Scripts\activate.bat" %ENV_NAME%
>>"%DESKTOP_LAUNCHER%" echo cd /d "%PROJECT_ROOT%"
>>"%DESKTOP_LAUNCHER%" echo python main.py %%*
if not exist "%DESKTOP_LAUNCHER%" (
    echo [install] WARNING: could not write the Desktop launcher.
    echo [install] Launch manually: activate '%ENV_NAME%' then run
    echo [install]   python "%PROJECT_ROOT%\main.py"
)

REM ---- 5. Optional SAM-HeLa checkpoint path -----------------------
echo.
echo [install] Optional: if you already have sam_hela\best.pt on disk,
echo [install] you can point the app at it now. Press Enter to skip --
echo [install] see hint below if you do.
set "MODEL_PATH="
set /p "MODEL_PATH=  Path to best.pt (or empty to skip): "
REM Strip quotes so Explorer's "Copy as path" (which quotes) works.
if defined MODEL_PATH set "MODEL_PATH=%MODEL_PATH:"=%"
if defined MODEL_PATH (
    call conda activate "%ENV_NAME%"
    python "%SCRIPT_DIR%configure_model.py" "%MODEL_PATH%" || (
        echo [install] ^(path not saved; you can set it later in the app settings^)
    )
) else (
    echo.
    echo [install] Skipped. Easiest way to add it later -- drop the file at:
    echo [install]   "%PROJECT_ROOT%\models\checkpoints\sam_hela\best.pt"
    echo [install] The app checks this exact path on startup; no further config needed.
    echo [install] Or register it in the app: Model menu -^> Add model...
)

echo.
echo [install] Done. Double-click EyeDataLabeller on your Desktop to launch.
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo [install] FAILED. See messages above.
echo [install] To report the problem: screenshot this window.
echo.
pause
endlocal
exit /b 1
