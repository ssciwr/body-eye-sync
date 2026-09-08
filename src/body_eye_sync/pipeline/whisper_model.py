"""Resolve Hugging Face Whisper checkpoints for faster-whisper."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from body_eye_sync.pipeline.model_cache import model_cache_dir

_HUB_MODEL_FILES = ["*.json", "*.safetensors", "*.bin", "*.txt"]


def _is_ctranslate2_model(path: Path) -> bool:
    import ctranslate2

    return ctranslate2.contains_model(str(path))


def _is_cached_conversion(path: Path) -> bool:
    return _is_ctranslate2_model(path) and (path / "tokenizer.json").is_file()


def _write_tokenizer(source: Path, converted: Path) -> None:
    """Write the source checkpoint's fast tokenizer in faster-whisper's format."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError(f"Whisper checkpoint has no fast tokenizer: {source}")
    backend.save(str(converted / "tokenizer.json"))


def _convert_transformers_checkpoint(source: Path, target: Path) -> Path:
    """Convert ``source`` into ``target`` without exposing a partial model."""
    from ctranslate2.converters import TransformersConverter

    if target.exists():
        shutil.rmtree(target)

    temporary_root = Path(tempfile.mkdtemp(prefix=".convert-", dir=target.parent))
    converted = temporary_root / "model"
    try:
        copy_files = [
            filename
            for filename in ("preprocessor_config.json", "tokenizer.json")
            if (source / filename).is_file()
        ]
        converter = TransformersConverter(
            str(source),
            copy_files=copy_files,
            load_as_float16=True,
            low_cpu_mem_usage=True,
        )
        converter.convert(str(converted), quantization="float16")
        if not (converted / "tokenizer.json").is_file():
            _write_tokenizer(source, converted)
        try:
            converted.rename(target)
        except FileExistsError:
            if not _is_cached_conversion(target):
                raise
        return target
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def download_whisper_model(model_name: str) -> Path:
    """Download a Hugging Face Whisper repository into the shared cache."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name,
            cache_dir=model_cache_dir() / "hugging-face",
            allow_patterns=_HUB_MODEL_FILES,
        )
    )


def resolve_whisper_model(model_name: str) -> str:
    """Return a CTranslate2 model path or a faster-whisper built-in name.

    Built-in names such as ``large-v3`` remain under faster-whisper's control.
    A Hugging Face repository is downloaded once and inspected with
    :func:`ctranslate2.contains_model`; native CTranslate2 weights are used as
    they are, while Transformers weights are converted into the shared cache.
    """
    if "/" not in model_name or Path(model_name).is_dir():
        return model_name

    models_dir = model_cache_dir()
    converted = models_dir / "ctranslate2" / model_name.replace("/", "--")
    if _is_cached_conversion(converted):
        return str(converted)

    source = download_whisper_model(model_name)
    if _is_ctranslate2_model(source):
        return str(source)

    converted.parent.mkdir(parents=True, exist_ok=True)
    return str(_convert_transformers_checkpoint(source, converted))


__all__ = ["download_whisper_model", "resolve_whisper_model"]
