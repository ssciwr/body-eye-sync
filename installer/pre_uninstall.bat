@echo off
REM Remove the Start-menu/desktop shortcut created in post_install.bat. Runs
REM while the env still exists, so python.exe and menuinst are available.
if exist "%PREFIX%\Menu\body-eye-sync.json" (
    "%PREFIX%\python.exe" -c "from menuinst.api import remove; remove(r'%PREFIX%\Menu\body-eye-sync.json', target_prefix=r'%PREFIX%', base_prefix=r'%PREFIX%')"
)
