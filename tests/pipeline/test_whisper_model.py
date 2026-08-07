from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import ctranslate2
import ctranslate2.converters
import huggingface_hub

from body_eye_sync.pipeline.whisper_model import resolve_whisper_model


def _mock_model_format(monkeypatch):
    monkeypatch.setattr(
        ctranslate2,
        "contains_model",
        lambda path: (
            (Path(path) / "model.bin").is_file()
            and (Path(path) / "config.json").is_file()
        ),
    )


def test_transformers_checkpoint_is_detected_and_converted_once(monkeypatch, tmp_path):
    _mock_model_format(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "preprocessor_config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    downloads = []
    conversions = []
    tokenizers = []

    def download(**options):
        downloads.append(options)
        return str(source)

    class Converter:
        def __init__(self, model_name, **options):
            conversions.append([model_name, options])

        def convert(self, output_dir, **options):
            conversions[-1].append(options)
            output = Path(output_dir)
            output.mkdir()
            (output / "model.bin").write_bytes(b"converted")
            (output / "config.json").write_text("{}")
            for filename in conversions[-1][1]["copy_files"]:
                shutil.copyfile(source / filename, output / filename)

    class BackendTokenizer:
        def save(self, path):
            Path(path).write_text("generated from source")

    class Tokenizer:
        backend_tokenizer = BackendTokenizer()

    def load_tokenizer(model_path, **options):
        tokenizers.append((model_path, options))
        return Tokenizer()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", Converter)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)),
    )
    monkeypatch.setattr(
        "body_eye_sync.pipeline.whisper_model.model_cache_dir", lambda: tmp_path
    )

    model_name = "primeline/whisper-large-v3-turbo-german"
    first = resolve_whisper_model(model_name)
    second = resolve_whisper_model(model_name)

    assert first == second
    assert Path(first, "model.bin").read_bytes() == b"converted"
    assert Path(first, "tokenizer.json").read_text() == "generated from source"
    assert not list(tmp_path.glob("ctranslate2/.convert-*"))
    assert downloads == [
        {
            "repo_id": model_name,
            "cache_dir": tmp_path / "hugging-face",
            "allow_patterns": ["*.json", "*.safetensors", "*.bin", "*.txt"],
        }
    ]
    assert conversions == [
        [
            str(source),
            {
                "copy_files": ["preprocessor_config.json"],
                "load_as_float16": True,
                "low_cpu_mem_usage": True,
            },
            {"quantization": "float16"},
        ]
    ]
    assert tokenizers == [(str(source), {"use_fast": True})]


def test_repository_tokenizer_is_kept(monkeypatch, tmp_path):
    _mock_model_format(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "preprocessor_config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("repository tokenizer")

    class Converter:
        def __init__(self, model_name, **options):
            self.source = Path(model_name)
            self.copy_files = options["copy_files"]

        def convert(self, output_dir, **options):
            output = Path(output_dir)
            output.mkdir()
            (output / "model.bin").write_bytes(b"converted")
            (output / "config.json").write_text("{}")
            for filename in self.copy_files:
                shutil.copyfile(self.source / filename, output / filename)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **options: source)
    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", Converter)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("tokenizer should have been copied")
                )
            )
        ),
    )
    monkeypatch.setattr(
        "body_eye_sync.pipeline.whisper_model.model_cache_dir", lambda: tmp_path
    )

    resolved = resolve_whisper_model("example/transformers-whisper")

    assert Path(resolved, "tokenizer.json").read_text() == "repository tokenizer"


def test_native_ctranslate2_repository_is_not_converted(monkeypatch, tmp_path):
    _mock_model_format(monkeypatch)
    source = tmp_path / "native"
    source.mkdir()
    (source / "model.bin").write_bytes(b"native")
    (source / "config.json").write_text("{}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **options: source)
    monkeypatch.setattr(
        ctranslate2.converters,
        "TransformersConverter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native model should not be converted")
        ),
    )
    monkeypatch.setattr(
        "body_eye_sync.pipeline.whisper_model.model_cache_dir", lambda: tmp_path
    )

    assert resolve_whisper_model("example/faster-whisper") == str(source)


def test_faster_whisper_alias_is_left_alone(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **options: (_ for _ in ()).throw(
            AssertionError("alias should be resolved by faster-whisper")
        ),
    )

    assert resolve_whisper_model("large-v3") == "large-v3"
