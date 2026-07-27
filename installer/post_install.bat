@echo off
REM Install the dependencies exported from uv.lock, selecting the CUDA 12.6
REM Torch wheels. uv verifies every hash available in the exported requirements;
REM the PyTorch index currently omits hashes for some Torchvision wheels. This is
REM the multi-GB download, and constructor shows little progress until it finishes.
REM uv comes from the conda-forge `uv` package listed in construct.yaml specs,
REM which installs it at this fixed location inside the prefix.
set "UV_EXE=%PREFIX%\Library\bin\uv.exe"
if not exist "%UV_EXE%" (
    echo ERROR: uv not found at "%UV_EXE%" - the installer payload is incomplete.
    exit /b 1
)
"%UV_EXE%" pip install --python "%PREFIX%\python.exe" --torch-backend cu126 -r "%PREFIX%\requirements-win.lock" || exit /b 1

REM Install the application itself; its dependencies are already satisfied above.
for %%f in ("%PREFIX%\body_eye_sync-*.whl") do (
    "%PREFIX%\python.exe" -m pip install --no-deps "%%f" || exit /b 1
)

REM Create the Start-menu/desktop shortcut via menuinst.
if not exist "%PREFIX%\Menu" mkdir "%PREFIX%\Menu"
copy /Y "%PREFIX%\menu.json" "%PREFIX%\Menu\body-eye-sync.json" >nul
"%PREFIX%\python.exe" -c "from menuinst.api import install; install(r'%PREFIX%\Menu\body-eye-sync.json', target_prefix=r'%PREFIX%', base_prefix=r'%PREFIX%')" || exit /b 1

REM Remove the installer payload (wheel + lockfile) now that it has been consumed.
del /q "%PREFIX%\body_eye_sync-*.whl" >nul 2>&1
del /q "%PREFIX%\requirements-win.lock" >nul 2>&1
