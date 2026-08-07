import locale
import os
import sys

import click

_MEDIA_FEATURE_PACK_MESSAGE = (
    "OpenCV requires Media Foundation (mfplat.dll), which is not installed on "
    "this system.\n\n"
    "Please install the Media Feature Pack and "
    "restart the application:\n\n"
    "Settings → Apps → Optional features → Add a feature "
    "→ Media Feature Pack"
)


def _ensure_standard_streams() -> None:
    """Ensure stdout/stderr exist (They aren't defined on Windows with pythonw.exe)."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
        original_name = f"__{name}__"
        if getattr(sys, original_name, None) is None:
            setattr(sys, original_name, getattr(sys, name))


_ensure_standard_streams()


def _media_foundation_missing() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        ctypes.WinDLL("mfplat.dll")
    except OSError:
        return True
    return False


def _use_c_numeric_locale() -> None:
    """Keep C number formatting after Qt has applied the system locale.

    Constructing a ``QApplication`` calls ``setlocale(LC_ALL, "")``. Where the
    system locale writes decimals with a comma, FFmpeg then rejects the filter
    options we build with a decimal point, such as ``atempo=0.8``. Qt formats
    numbers through ``QLocale``, so restoring this one category costs nothing.
    """
    locale.setlocale(locale.LC_NUMERIC, "C")


@click.command()
@click.argument("experiment", required=False)
@click.version_option(package_name="body-eye-sync", prog_name="body-eye-sync")
def main(experiment):
    from qtpy.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    _use_c_numeric_locale()

    if _media_foundation_missing():
        QMessageBox.critical(
            None, "Missing Windows component", _MEDIA_FEATURE_PACK_MESSAGE
        )
        return

    from body_eye_sync.gui import MainWindow
    from body_eye_sync.gui.autoupdate import AutoUpdater

    window = MainWindow()
    window.show()
    if experiment is not None:
        window.load_experiment(experiment)

    updater = AutoUpdater(window)
    updater.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
