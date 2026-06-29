@echo off
REM Unregister the Start-menu/desktop shortcut
if exist "%PREFIX%\Menu\body-eye-sync.json" (
    "%PREFIX%\python.exe" -c "from menuinst.api import remove; remove(r'%PREFIX%\Menu\body-eye-sync.json', target_prefix=r'%PREFIX%', base_prefix=r'%PREFIX%')"
)

REM Remove the leftover folder/file
del /q "%PREFIX%\menu.json" >nul 2>&1
rmdir /s /q "%PREFIX%\Menu" >nul 2>&1
