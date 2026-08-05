"""The tab base class, and the placeholder tabs that are nothing more than it."""

import pytest
from qtpy.QtWidgets import QLabel

from body_eye_sync.experiment.config import ExperimentConfig
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs import TAB_TYPES
from body_eye_sync.gui.tabs.base import BaseTab, PlaceholderTab
from body_eye_sync.gui.tabs.data_export import DataExportTab
from body_eye_sync.gui.tabs.post_processing import PostProcessingTab

PLACEHOLDER_TABS = [
    PostProcessingTab,
    DataExportTab,
]


@pytest.fixture
def experiment():
    return Experiment(ExperimentConfig())


def test_every_tab_has_a_title():
    assert all(tab_type.title for tab_type in TAB_TYPES)


def test_a_tab_holds_the_experiment_it_is_given(qtbot, experiment):
    tab = BaseTab(experiment)
    qtbot.addWidget(tab)

    assert tab.experiment is experiment


def test_set_experiment_swaps_it_and_refreshes(qtbot, experiment):
    class _Tab(BaseTab):
        refreshed = 0

        def refresh(self):
            self.refreshed += 1

    tab = _Tab(experiment)
    qtbot.addWidget(tab)
    other = Experiment(ExperimentConfig())

    tab.set_experiment(other)

    assert tab.experiment is other
    assert tab.refreshed == 1


@pytest.mark.parametrize("tab_type", PLACEHOLDER_TABS, ids=lambda t: t.__name__)
def test_placeholder_tabs_say_so(qtbot, experiment, tab_type):
    tab = tab_type(experiment)
    qtbot.addWidget(tab)

    assert issubclass(tab_type, PlaceholderTab)
    label = tab.findChild(QLabel)
    assert label.text() == f"{tab_type.title} is not implemented yet"
