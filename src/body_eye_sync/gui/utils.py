from collections.abc import Iterable

from qtpy.QtGui import QColor

_PALETTE = (
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
)

_RECORDING_COLOR_IDS = (0, 6, 3, 9, 18, 10, 5, 7, 13, 15, 1, 8, 17, 19)


def get_color(object_id: int) -> QColor:
    return QColor("#" + _PALETTE[int(object_id) % len(_PALETTE)])


def recording_colors(recording_ids: Iterable[str]) -> dict[str, QColor]:
    """Assign the shared recording palette in experiment input order."""
    return {
        recording_id: get_color(_RECORDING_COLOR_IDS[index % len(_RECORDING_COLOR_IDS)])
        for index, recording_id in enumerate(recording_ids)
    }
