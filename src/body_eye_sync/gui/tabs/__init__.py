"""The tabs the main window is made of, in the order they are shown.

Each tab is one stage of working with an experiment, and each lives in its own
module. Adding a stage means writing a :class:`~body_eye_sync.gui.tabs.base.BaseTab`
subclass and listing it in :data:`TAB_TYPES`; the window needs no changes.
"""

from __future__ import annotations

from body_eye_sync.gui.tabs.alignment import AlignmentTab
from body_eye_sync.gui.tabs.audio_processing import AudioProcessingTab
from body_eye_sync.gui.tabs.base import BaseTab, PlaceholderTab
from body_eye_sync.gui.tabs.data_export import DataExportTab
from body_eye_sync.gui.tabs.input_files import InputFilesTab
from body_eye_sync.gui.tabs.post_processing import PostProcessingTab
from body_eye_sync.gui.tabs.video_processing import VideoProcessingTab

# The tabs in the order that they should be displayed
TAB_TYPES: tuple[type[BaseTab], ...] = (
    InputFilesTab,
    AlignmentTab,
    VideoProcessingTab,
    AudioProcessingTab,
    PostProcessingTab,
    DataExportTab,
)

__all__ = [
    "TAB_TYPES",
    "AlignmentTab",
    "AudioProcessingTab",
    "BaseTab",
    "DataExportTab",
    "InputFilesTab",
    "PlaceholderTab",
    "PostProcessingTab",
    "VideoProcessingTab",
]
