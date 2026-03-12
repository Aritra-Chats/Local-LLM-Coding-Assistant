@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: run_sentinel.bat  —  Launch the Sentinel Local AI Development Assistant
::
:: Usage
:: -----
::   run_sentinel.bat                       Start a new session
::   run_sentinel.bat --resume SESSION_ID   Resume a saved session
::   run_sentinel.bat --project C:\my\code  Assist a specific project
::   run_sentinel.bat --mode minimal        Override hardware profile
::   run_sentinel.bat --no-bootstrap        Skip the first-launch setup
::
:: All extra arguments are forwarded directly to main.py.
:: ─────────────────────────────────────────────────────────────────────────────

:: Capture the directory sentinel was launched FROM before we cd away.
set "SENTINEL_PROJECT_DIR=%CD%"

:: Move to the directory that contains this .bat file, then project root.
cd /d "%~dp0"

:: Paths
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "ENTRY_POINT=%~dp0main.py"

:: ── Pre-flight checks ────────────────────────────────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Expected: %VENV_PYTHON%
    echo.
    echo  Create it with:  python -m venv .venv
    echo  Then install deps:  .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "%ENTRY_POINT%" (
    echo.
    echo  ERROR: Entry point not found: %ENTRY_POINT%
    echo.
    pause
    exit /b 1
)

:: ── Optional: set Sentinel home directory ────────────────────────────────────
:: Uncomment and adjust the line below to override where sessions/metrics
:: are stored.  Default is %%USERPROFILE%%\.sentinel
::
:: set "SENTINEL_HOME=C:\sentinel-data"

:: ── Launch ───────────────────────────────────────────────────────────────────
echo.
echo  Starting Sentinel...
echo.
"%VENV_PYTHON%" "%ENTRY_POINT%" %*

:: ── On non-zero exit, pause so the user can read the error ───────────────────
if errorlevel 1 (
    echo.
    echo  Sentinel exited with an error (code %errorlevel%).
    echo  Check the output above for details.
    echo.
    pause
)
