"""Post processing tab: what is derived from the per-input pipeline results."""

from __future__ import annotations

from body_eye_sync.gui.tabs.base import PlaceholderTab


class PostProcessingTab(PlaceholderTab):
    """Combine the per-input results into experiment-level results."""

    title = "Post processing"
