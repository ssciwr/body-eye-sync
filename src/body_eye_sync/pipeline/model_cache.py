"""Shared on-disk cache for downloaded model weights.

A single cross-platform cache directory that every pipeline step routes its
model downloads into, rather than scattering them across per-library defaults
(Ultralytics' working directory, InsightFace's ``~/.insightface``).
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_path


def model_cache_dir() -> Path:
    """Directory all downloaded model weights are cached in (cross-platform)."""
    return user_cache_path("body-eye-sync", "SSC") / "models"


def cached_model_path(model_ref: str | Path) -> str:
    """Route a bare weights filename into the shared model cache.

    Given only a bare name (e.g. ``yolo26m.pt``), Ultralytics downloads the
    weights into the current working directory. Rewriting it to an absolute path
    under :func:`model_cache_dir` makes the download land in the shared cache
    instead. An explicit path (one with a directory part) or an already-existing
    file is returned unchanged.
    """
    path = Path(model_ref)
    if path.parent != Path(".") or path.exists():
        return str(model_ref)
    cache_dir = model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / path.name)
