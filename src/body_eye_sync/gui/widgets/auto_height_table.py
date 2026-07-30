"""A table that grows with its rows instead of scrolling inside itself."""

from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSizePolicy,
    QTableWidget,
)


class AutoHeightTable(QTableWidget):
    """A table sized to exactly fit its header and rows.

    Several tabs stack tables down a scrolling page. Each one should be as tall
    as it needs to be and let the page scroll, rather than scrolling within a
    height nobody chose, so the height follows the row count and the table's
    own scrollbar stays out of the way.
    """

    def __init__(self, headers: Sequence[str]) -> None:
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def fit_to_rows(self) -> None:
        """Make the table exactly as tall as its header and rows."""
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        for row in range(self.rowCount()):
            height += self.rowHeight(row)
        self.setFixedHeight(height)
