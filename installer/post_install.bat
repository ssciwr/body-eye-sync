@echo off
REM Install boxmot (not on conda) using pip with --no-deps - pinned version to match pyproject.toml and the boxmot deps mirrored in construct.yaml
"%PREFIX%\python.exe" -m pip install --no-deps boxmot==21.0.0 || exit /b 1

REM Install our package, globbing the wheel to avoid hard-coding the version here
for %%f in ("%PREFIX%\body_eye_sync-*.whl") do (
    "%PREFIX%\python.exe" -m pip install --no-deps "%%f" || exit /b 1
)

REM Create the Start-menu/desktop shortcut
if not exist "%PREFIX%\Menu" mkdir "%PREFIX%\Menu"
copy /Y "%PREFIX%\menu.json" "%PREFIX%\Menu\body-eye-sync.json" >nul
"%PREFIX%\python.exe" -c "from menuinst.api import install; install(r'%PREFIX%\Menu\body-eye-sync.json', target_prefix=r'%PREFIX%', base_prefix=r'%PREFIX%')" || exit /b 1
