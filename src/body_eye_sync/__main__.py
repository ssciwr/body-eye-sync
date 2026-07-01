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
    """Provide writable streams when Windows starts the GUI via pythonw.exe."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
        original_name = f"__{name}__"
        if getattr(sys, original_name, None) is None:
            setattr(sys, original_name, getattr(sys, name))


_ensure_standard_streams()


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


@click.command()
@click.argument("experiment", required=False)
@click.version_option(package_name="body-eye-sync", prog_name="body-eye-sync")
def main(experiment):
    """Launch the GUI, optionally opening the EXPERIMENT folder on startup."""
    from qtpy.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)

    if _media_foundation_missing():
        QMessageBox.critical(
            None, "Missing Windows component", _MEDIA_FEATURE_PACK_MESSAGE
        )
        return

    from body_eye_sync.gui import MainWindow
    from body_eye_sync.gui.autoupdate import AutoUpdater

    window = MainWindow()
    window.show()

    # A folder passed on the command line is opened as an experiment; any problem
    # (missing config, bad video path) is surfaced in the window, not on stderr.
    if experiment is not None:
        window.load_experiment(experiment)

    # Check GitHub for a newer version and, if found, offer to update.
    updater = AutoUpdater(window)
    updater.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
