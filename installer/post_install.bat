@echo off
REM Add the two pure-Python packages that are not on conda-forge. --no-deps is
REM safe because every dependency is already installed from conda (see specs in
REM construct.yaml). boxmot is pulled from PyPI; our own package is the wheel
REM bundled into the install prefix via extra_files.
"%PREFIX%\python.exe" -m pip install --no-deps "%PREFIX%\body_eye_sync-0.0.1-py3-none-any.whl" || exit /b 1
"%PREFIX%\python.exe" -m pip install --no-deps boxmot || exit /b 1

REM Create the Start-menu/desktop shortcut. menuinst's `activate: true` makes
REM the shortcut launch through the conda env so the GUI finds its torch/Qt DLLs.
if not exist "%PREFIX%\Menu" mkdir "%PREFIX%\Menu"
copy /Y "%PREFIX%\menu.json" "%PREFIX%\Menu\body-eye-sync.json" >nul
"%PREFIX%\python.exe" -c "from menuinst.api import install; install(r'%PREFIX%\Menu\body-eye-sync.json', target_prefix=r'%PREFIX%', base_prefix=r'%PREFIX%')" || exit /b 1
