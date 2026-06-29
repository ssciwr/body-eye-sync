@echo off
REM Install the pinned, hash-verified dependencies (including CUDA torch) from the
REM lockfile shipped inside the installer. This is the multi-GB download; the
REM constructor window shows little progress while it runs (output is shown only
REM once this finishes).
"%PREFIX%\python.exe" -m pip install --no-warn-script-location --require-hashes -r "%PREFIX%\requirements-win.lock" || exit /b 1

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
