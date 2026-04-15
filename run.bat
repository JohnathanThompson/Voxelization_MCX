@echo off
echo =============================================
echo  Running SimVascular CFD Pipeline
echo =============================================
echo.

for /f "delims=" %%i in ('wsl -d Ubuntu wslpath -a "%~dp0."') do set WSL_DIR=%%i
echo Project directory: %WSL_DIR%
echo.

echo Running pipeline...
echo.

wsl -d Ubuntu bash -c "export LD_LIBRARY_PATH=/usr/local/sv/simvascular/2025-12-21/lib:/usr/local/sv/simvascular/2025-12-21/lib/plugins:/usr/local/sv/simvascular/2025-12-21/bin:/usr/local/sv/simvascular/2025-12-21/bin/assetimporters:/usr/local/sv/simvascular/2025-12-21/bin/designer:/usr/local/sv/simvascular/2025-12-21/bin/egldeviceintegrations:/usr/local/sv/simvascular/2025-12-21/bin/generic:/usr/local/sv/simvascular/2025-12-21/bin/geometryloaders:/usr/local/sv/simvascular/2025-12-21/bin/iconengines:/usr/local/sv/simvascular/2025-12-21/bin/imageformats:/usr/local/sv/simvascular/2025-12-21/bin/networkinformation:/usr/local/sv/simvascular/2025-12-21/bin/opcua:/usr/local/sv/simvascular/2025-12-21/bin/platforminputcontexts:/usr/local/sv/simvascular/2025-12-21/bin/platforms:/usr/local/sv/simvascular/2025-12-21/bin/platformthemes:/usr/local/sv/simvascular/2025-12-21/bin/position:/usr/local/sv/simvascular/2025-12-21/bin/printsupport:/usr/local/sv/simvascular/2025-12-21/bin/qmllint:/usr/local/sv/simvascular/2025-12-21/bin/qmltooling:/usr/local/sv/simvascular/2025-12-21/bin/renderers:/usr/local/sv/simvascular/2025-12-21/bin/renderplugins:/usr/local/sv/simvascular/2025-12-21/bin/sceneparsers:/usr/local/sv/simvascular/2025-12-21/bin/scxmldatamodel:/usr/local/sv/simvascular/2025-12-21/bin/sqldrivers:/usr/local/sv/simvascular/2025-12-21/bin/tls:/usr/local/sv/simvascular/2025-12-21/bin/wayland-decoration-client:/usr/local/sv/simvascular/2025-12-21/bin/wayland-graphics-integration-client:/usr/local/sv/simvascular/2025-12-21/bin/wayland-graphics-integration-server:/usr/local/sv/simvascular/2025-12-21/bin/wayland-shell-integration:/usr/local/sv/simvascular/2025-12-21/bin/webview:/usr/local/sv/simvascular/2025-12-21/bin/xcbglintegrations:/usr/local/sv/simvascular/2025-12-21/svExternals/bin:/usr/local/sv/simvascular/2025-12-21/svExternals/bin/MitkCore:/usr/local/sv/simvascular/2025-12-21/svExternals/bin/MitkDICOM:/usr/local/sv/simvascular/2025-12-21/svExternals/bin/MitkModelFit:/usr/local/sv/simvascular/2025-12-21/svExternals/lib:/usr/local/sv/simvascular/2025-12-21/svExternals/lib/MitkCore:/usr/local/sv/simvascular/2025-12-21/svExternals/lib/MitkDICOM:/usr/local/sv/simvascular/2025-12-21/svExternals/lib/MitkModelFit:/usr/local/sv/simvascular/2025-12-21/svExternals/lib/plugins:/usr/local/sv/simvascular/2025-12-21/svExternals/lib/python3.11/lib-dynload:$LD_LIBRARY_PATH && /usr/local/sv/simvascular/2025-12-21/bin/simvascular --python -- '%WSL_DIR%/Pipeline_fast.py'"

echo.
echo =============================================
if %ERRORLEVEL% == 0 (
    echo  Done! Results saved to TestPy/sim/
    echo  Open result_*.vtu files in ParaView
) else (
    echo  ERROR: Simulation failed. Check output above.
)
echo =============================================
pause
