@echo off
setlocal

rem ---------------------------------------------------------------------------
rem Sentinel launcher for Windows
rem
rem PATH-safe: %~dp0 always resolves to the directory that contains THIS .bat
rem file, even when the sentinel folder is added to PATH system-wide.
rem %CD% is captured BEFORE any pushd so it reflects the directory where the
rem user ran "sentinel" — that becomes the project root passed to main.py.
rem ---------------------------------------------------------------------------

rem Capture caller CWD first (before pushd changes it)
set "ORIGINAL_CWD=%CD%"
set "SENTINEL_DIR=%~dp0"

rem Strip trailing backslash so paths compose cleanly
if "%SENTINEL_DIR:~-1%"=="\" set "SENTINEL_DIR=%SENTINEL_DIR:~0,-1%"

set "VENV_DIR=%SENTINEL_DIR%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%SENTINEL_DIR%\requirements.txt"
set "MAIN_FILE=%SENTINEL_DIR%\main.py"
set "BASE_PYTHON="
set "BASE_PYTHON_ARGS="

echo Sentinel launcher (windows)
echo   Installation : %SENTINEL_DIR%
echo   Project root : %ORIGINAL_CWD%

rem Work from the installation dir for dependency resolution
pushd "%SENTINEL_DIR%" >nul

call :ResolvePython
if not defined BASE_PYTHON (
    echo Python is not installed.
    echo Attempting to install Python with winget...
    call :InstallPython
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
    call :ResolvePython
)

if not defined BASE_PYTHON (
    echo Python installation completed, but the launcher could not locate the executable.
    echo Open a new PowerShell window and run sentinel again.
    popd >nul
    exit /b 1
)

call :EnsureOllama
if errorlevel 1 (
    popd >nul
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python virtual environment...
    "%BASE_PYTHON%" %BASE_PYTHON_ARGS% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        popd >nul
        exit /b 1
    )
)

if not exist "%PYTHON_EXE%" (
    echo Virtual environment python was not found.
    popd >nul
    exit /b 1
)

echo Installing or updating Python requirements...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo Failed to upgrade pip.
    popd >nul
    exit /b 1
)

"%PYTHON_EXE%" -m pip install -r "%REQ_FILE%" --quiet
if errorlevel 1 (
    echo Failed to install Python requirements.
    popd >nul
    exit /b 1
)

echo Starting Sentinel...
"%PYTHON_EXE%" "%MAIN_FILE%" --project "%ORIGINAL_CWD%" %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
exit /b %EXIT_CODE%

:ResolvePython
set "BASE_PYTHON="
set "BASE_PYTHON_ARGS="

where python >nul 2>nul
if not errorlevel 1 (
    set "BASE_PYTHON=python"
    exit /b 0
)

where py >nul 2>nul
if not errorlevel 1 (
    set "BASE_PYTHON=py"
    set "BASE_PYTHON_ARGS=-3"
    exit /b 0
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python313\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
) do (
    if exist %%~P (
        set "BASE_PYTHON=%%~P"
        exit /b 0
    )
)

exit /b 0

:InstallPython
where winget >nul 2>nul
if errorlevel 1 (
    echo winget was not found. Attempting to download and install via PowerShell...
    powershell -NoProfile -Command "$out = Join-Path $env:TEMP 'winget.msixbundle'; Invoke-WebRequest -Uri 'https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle' -OutFile $out; Add-AppxPackage -ForceApplicationShutdown -ForceUpdateFromAnyVersion -Path $out" 2>nul
    if errorlevel 1 (
        echo Automated winget installation failed. Please install winget or App Installer from the Microsoft Store, then rerun sentinel.
        exit /b 1
    )
)

winget install -e --id Python.Python.3.12 -h
if errorlevel 1 (
    echo Python installation failed. Install Python manually, then run sentinel again.
    exit /b 1
)

exit /b 0

:EnsureOllama
where ollama >nul 2>nul
if not errorlevel 1 (
    exit /b 0
)

echo Ollama is not installed.
echo Attempting to install Ollama with winget...
where winget >nul 2>nul
if errorlevel 1 (
    echo winget is not installed. Attempting to download and install via PowerShell...
    powershell -NoProfile -Command "$out = Join-Path $env:TEMP 'winget.msixbundle'; Invoke-WebRequest -Uri 'https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle' -OutFile $out; Add-AppxPackage -ForceApplicationShutdown -ForceUpdateFromAnyVersion -Path $out" 2>nul
    if errorlevel 1 (
        echo Automated winget installation failed. Please install winget/App Installer manually, then rerun sentinel.
        exit /b 1
    )
)

winget install -e --id Ollama.Ollama -h
if errorlevel 1 (
    echo Ollama installation failed. Install Ollama manually, then run sentinel again.
    exit /b 1
)

exit /b 0
