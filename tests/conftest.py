from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory containing test data files (videos, etc.)."""
    return Path(__file__).parent / "data"
