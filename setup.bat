@echo off
setlocal EnableExtensions EnableDelayedExpansion
echo =============================================
echo  SimVascular One-Time Setup
echo =============================================
echo.

REM -----------------------------------------------
REM PHASE 1: Check if WSL2 kernel is available
REM -----------------------------------------------
wsl --status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WSL2 is not installed. Installing now...
    echo A restart will be required after this step.
    echo Run setup.bat again after restarting.
    echo.
    wsl --install --no-distribution
    echo.
    echo =============================================
    echo  RESTART YOUR COMPUTER
    echo  Then double-click setup.bat again.
    echo =============================================
    pause
    exit /b
)
echo [OK] WSL2 kernel is installed.
echo.

REM -----------------------------------------------
REM PHASE 2: Check if Ubuntu is installed and working
REM -----------------------------------------------
wsl -d Ubuntu bash -c "echo ok" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ubuntu is not set up yet. Installing now...
    echo A Ubuntu terminal will open asking for a username
    echo and password. Fill those in, then close that window
    echo and double-click setup.bat again.
    echo.
    wsl --install -d Ubuntu
    echo.
    echo =============================================
    echo  After setting your Ubuntu username/password,
    echo  close the Ubuntu window and run setup.bat again.
    echo =============================================
    pause
    exit /b
)

REM Print Ubuntu version so user can confirm it matches their .deb download
echo [OK] Ubuntu is installed. Version:
wsl -d Ubuntu bash -c "lsb_release -d"
echo.

REM -----------------------------------------------
REM PHASE 3: Check if SimVascular, svMultiPhysics,
REM and mpiexec are installed
REM -----------------------------------------------
set SIMVASCULAR_INSTALLED=0
wsl -d Ubuntu bash -c "dpkg -l | grep -i simvascular" >nul 2>&1
if not errorlevel 1 (
    set SIMVASCULAR_INSTALLED=1
    echo [OK] SimVascular is already installed.
)
echo.

set SVMULTIPHYSICS_INSTALLED=0
wsl -d Ubuntu bash -c "find /usr/local/sv /opt -path '*/bin/svmultiphysics' -type f 2>/dev/null | grep -q svmultiphysics"
if not errorlevel 1 (
    set SVMULTIPHYSICS_INSTALLED=1
    echo [OK] svMultiPhysics is already installed.
)
echo.

set MPIEXEC_INSTALLED=0
wsl -d Ubuntu bash -c "command -v mpiexec >/dev/null 2>&1"
if not errorlevel 1 (
    set MPIEXEC_INSTALLED=1
    echo [OK] mpiexec is already installed in WSL.
)
echo.

if "!SIMVASCULAR_INSTALLED!"=="1" if "!SVMULTIPHYSICS_INSTALLED!"=="1" if "!MPIEXEC_INSTALLED!"=="1" (
    echo Nothing to do - you can run run.bat now.
    pause
    exit /b
)

REM -----------------------------------------------
REM PHASE 4: Check for manually placed .deb installer
REM -----------------------------------------------
if "!SIMVASCULAR_INSTALLED!"=="0" (
    if not exist "%~dp0sv_install.deb" (
        echo =============================================
        echo  ACTION REQUIRED - Manual Download Needed
        echo =============================================
        echo.
        echo  SimVascular must be downloaded manually.
        echo  Please do the following:
        echo.
        echo  1. Open this link in your browser:
        echo     https://simtk.org/projects/simvascular
        echo.
        echo  2. Click "Downloads" and download the
        echo     .deb installer for Ubuntu 24.04
        echo.
        echo  3. Rename the downloaded file to:
        echo     sv_install.deb
        echo.
        echo  4. Place sv_install.deb in this folder:
        echo     %~dp0
        echo.
        echo  5. Double-click setup.bat again
        echo =============================================
        pause
        exit /b
    )
    echo [OK] Installer found: sv_install.deb
    echo.
)

if "!SVMULTIPHYSICS_INSTALLED!"=="0" (
    if not exist "%~dp0svMultiPhysics.deb" (
        echo =============================================
        echo  ACTION REQUIRED - Manual Download Needed
        echo =============================================
        echo.
        echo  svMultiPhysics must be downloaded manually.
        echo  Please do the following:
        echo.
        echo  1. Open this link in your browser:
        echo     https://simtk.org/projects/simvascular
        echo.
        echo  2. Click "Downloads" and download the
        echo     svMultiPhysics .deb installer for Ubuntu 24.04
        echo.
        echo  3. Rename the downloaded file to:
        echo     svMultiPhysics.deb
        echo.
        echo  4. Place svMultiPhysics.deb in this folder:
        echo     %~dp0
        echo.
        echo  5. Double-click setup.bat again
        echo =============================================
        pause
        exit /b
    )
    echo [OK] Installer found: svMultiPhysics.deb
    echo.
)

REM -----------------------------------------------
REM PHASE 5: Install dependencies
REM libgl1-mesa-glx was removed in Ubuntu 24.04
REM and replaced with libgl1
REM -----------------------------------------------
echo Step 1: Installing WSL dependencies, including OpenMPI/mpiexec...
wsl -d Ubuntu bash -c "sudo apt-get update -qq && sudo apt-get install -y -qq openmpi-bin libopenmpi-dev libgl1 libglib2.0-0 libxrender1 libxext6 libegl1 libgles2"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies.
    echo Make sure your Ubuntu has internet access.
    pause
    exit /b
)
echo [OK] Dependencies installed.
echo.

echo Step 2: Verifying mpiexec inside WSL...
wsl -d Ubuntu bash -c "command -v mpiexec >/dev/null 2>&1"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: OpenMPI was installed, but mpiexec is still unavailable in WSL.
    echo Check your Ubuntu package state and try again.
    pause
    exit /b
)
echo [OK] mpiexec is available in WSL.
echo.

REM -----------------------------------------------
REM PHASE 6: Copy .deb into WSL and install it
REM -----------------------------------------------
set WSL_PROJ_DIR=

if "!SIMVASCULAR_INSTALLED!"=="0" if not defined WSL_PROJ_DIR (
    for /f "delims=" %%i in ('wsl wslpath -a "%~dp0."') do set WSL_PROJ_DIR=%%i
)

if "!SVMULTIPHYSICS_INSTALLED!"=="0" if not defined WSL_PROJ_DIR (
    for /f "delims=" %%i in ('wsl wslpath -a "%~dp0."') do set WSL_PROJ_DIR=%%i
)

if defined WSL_PROJ_DIR (
    echo [OK] WSL project path: !WSL_PROJ_DIR!
    echo.
)

wsl -d Ubuntu bash -c "dpkg -l | grep -i simvascular" >nul 2>&1
if not errorlevel 1 (
    echo [OK] Skipping SimVascular install.
    echo.
    goto after_simvascular_install
)

echo Step 3: Copying SimVascular installer into WSL...

wsl -d Ubuntu -- cp "!WSL_PROJ_DIR!sv_install.deb" /tmp/sv_install.deb
if errorlevel 1 (
    echo ERROR: Failed to copy sv_install.deb into WSL.
    pause
    exit /b
)
echo [OK] Installer copied.
echo.

echo Step 4: Installing SimVascular .deb package...
echo (This may take a few minutes)
echo.
set SIMVASCULAR_INSTALL_OK=0
wsl -d Ubuntu -- sudo dpkg -i /tmp/sv_install.deb
if errorlevel 1 (
    echo SimVascular requested additional packages. Repairing dependencies...
    wsl -d Ubuntu -- sudo apt-get install -f -y
    if errorlevel 1 (
        echo ERROR: Failed to repair SimVascular dependencies.
        pause
        exit /b
    )
    wsl -d Ubuntu -- sudo dpkg -i /tmp/sv_install.deb
)
if not errorlevel 1 set SIMVASCULAR_INSTALL_OK=1
wsl -d Ubuntu -- rm -f /tmp/sv_install.deb
if "!SIMVASCULAR_INSTALL_OK!" NEQ "1" (
    echo ERROR: SimVascular installation failed.
    echo Check the output above for details.
    pause
    exit /b
)
echo [OK] SimVascular package installed.
echo.

:after_simvascular_install

wsl -d Ubuntu bash -c "find /usr/local/sv /opt -path '*/bin/svmultiphysics' -type f 2>/dev/null | grep -q svmultiphysics"
if not errorlevel 1 (
    echo [OK] Skipping svMultiPhysics install.
    echo.
    goto after_svmultiphysics_install
)

echo Step 5: Copying svMultiPhysics installer into WSL...

wsl -d Ubuntu -- cp "!WSL_PROJ_DIR!svMultiPhysics.deb" /tmp/svMultiPhysics.deb
if errorlevel 1 (
    echo ERROR: Failed to copy svMultiPhysics.deb into WSL.
    pause
    exit /b
)
echo [OK] svMultiPhysics installer copied.
echo.

echo Step 6: Installing svMultiPhysics .deb package...
echo (This may take a few minutes)
echo.
set SVMULTIPHYSICS_INSTALL_OK=0
wsl -d Ubuntu -- sudo dpkg -i /tmp/svMultiPhysics.deb
if errorlevel 1 (
    echo svMultiPhysics requested additional packages. Repairing dependencies...
    wsl -d Ubuntu -- sudo apt-get install -f -y
    if errorlevel 1 (
        echo ERROR: Failed to repair svMultiPhysics dependencies.
        pause
        exit /b
    )
    wsl -d Ubuntu -- sudo dpkg -i /tmp/svMultiPhysics.deb
)
if not errorlevel 1 set SVMULTIPHYSICS_INSTALL_OK=1
wsl -d Ubuntu -- rm -f /tmp/svMultiPhysics.deb
if "!SVMULTIPHYSICS_INSTALL_OK!" NEQ "1" (
    echo ERROR: svMultiPhysics installation failed.
    echo Check the output above for details.
    pause
    exit /b
)
echo [OK] svMultiPhysics package installed.
echo.

:after_svmultiphysics_install

REM -----------------------------------------------
REM PHASE 7: Verify installation state
REM -----------------------------------------------
wsl -d Ubuntu bash -c "dpkg -l | grep -i simvascular" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: SimVascular is still not installed in WSL.
    pause
    exit /b
)

wsl -d Ubuntu bash -c "find /usr/local/sv /opt -path '*/bin/svmultiphysics' -type f 2>/dev/null | grep -q svmultiphysics"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: svMultiPhysics is still not installed in WSL.
    pause
    exit /b
)

REM -----------------------------------------------
REM PHASE 8: Find where SimVascular was installed
REM -----------------------------------------------
echo.
echo Step 7: Finding SimVascular and svMultiPhysics install locations...
wsl -d Ubuntu bash -c "find /usr -name 'simvascular' -type f 2>/dev/null; find /opt -name 'simvascular' -type f 2>/dev/null; find /usr/local/sv /opt -path '*/bin/svmultiphysics' -type f 2>/dev/null"

wsl -d Ubuntu bash -c "dpkg -l | grep -i simvascular >/dev/null 2>&1 && find /usr/local/sv /opt -path '*/bin/svmultiphysics' -type f 2>/dev/null | grep -q svmultiphysics"
if %ERRORLEVEL% == 0 (
    echo.
    echo =============================================
    echo  [OK] SimVascular, svMultiPhysics,
    echo  and mpiexec are installed successfully!
    echo.
    echo  NOTE: Check the paths printed above and
    echo  update run.bat or Pipeline_fast.py if the
    echo  installed paths differ from your expected
    echo  /usr/local/sv locations.
    echo.
    echo  Double-click run.bat to run your simulation.
    echo =============================================
) else (
    echo.
    echo =============================================
    echo  WARNING: Could not verify all installs.
    echo  Check the output above for the install paths
    echo  and update your scripts accordingly.
    echo =============================================
)
pause
