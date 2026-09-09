"""Read the JSON files a recording folder holds."""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path

import orjson


def json_file(path: Path) -> dict:
    """Parse a JSON file, or return an empty dict if it is not there."""
    return orjson.loads(path.read_bytes()) if path.is_file() else {}


def json_lines(path: Path) -> Iterator[dict]:
    """Parse a gzipped file holding one JSON record per line."""
    with gzip.open(path, "rb") as stream:
        for line in stream:
            yield orjson.loads(line)
