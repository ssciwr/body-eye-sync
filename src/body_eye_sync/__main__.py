import faulthandler
import sys
import traceback
from pathlib import Path

#: Startup errors are written here. The released app runs under pythonw.exe (no
#: console), so without this an import/DLL/Qt failure would vanish silently.
ERROR_LOG = Path.home() / "body-eye-sync-error.log"

_MEDIA_FEATURE_PACK_MESSAGE = (
    "OpenCV requires Media Foundation (mfplat.dll), which is not installed on "
    "this system.\n\n"
    "This is common on Windows 'N' editions. Install the Media Feature Pack and "
    "restart the application:\n\n"
    "Settings → Apps → Optional features → Add a feature "
    "→ Media Feature Pack"
)


def _register_conda_dll_dirs():
    """Make conda's native DLLs (opencv, ...) findable.

    Python 3.8+ ignores PATH for extension-module dependencies, so the
    launcher's PATH activation isn't enough; register the conda lib dirs
    explicitly so cv2 etc. load regardless of how the app was started.
    """
    if sys.platform != "win32":
        return
    import os

    for sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin"):
        path = Path(sys.prefix) / sub
        if path.is_dir():
            os.add_dll_directory(str(path))


def _media_foundation_missing() -> bool:
    """True on Windows when Media Foundation (needed by OpenCV) is unavailable."""
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        ctypes.WinDLL("mfplat.dll")
    except OSError:
        return True
    return False


def main():
    # faulthandler catches hard crashes (e.g. a failing native DLL or a Qt
    # platform-plugin abort) that don't raise a normal Python exception.
    log_file = ERROR_LOG.open("w")
    faulthandler.enable(log_file)
    try:
        _register_conda_dll_dirs()

        from qtpy.QtWidgets import QApplication, QMessageBox

        app = QApplication(sys.argv)

        # Check before importing the gui (which imports OpenCV) so the user gets
        # a clear dialog instead of a silent crash.
        if _media_foundation_missing():
            QMessageBox.critical(
                None, "Missing Windows component", _MEDIA_FEATURE_PACK_MESSAGE
            )
            return

        from body_eye_sync.gui import MainWindow

        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(file=log_file)
        log_file.flush()
        raise


if __name__ == "__main__":
    main()
