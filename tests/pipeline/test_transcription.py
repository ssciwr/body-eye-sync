from pathlib import Path

from body_eye_sync.pipeline.transcription import (
    SEGMENT_COLUMNS,
    WORD_COLUMNS,
    TranscriptSegment,
    Word,
    transcribe,
    transcript_to_dataframes,
)

CI_MODEL = "tiny"


def _segment(start, end, *words):
    return TranscriptSegment(start, end, " ".join(w.word for w in words), list(words))


def test_segments_are_numbered_in_the_order_they_were_spoken():
    transcript = [
        _segment(5.0, 6.0, Word(5.1, 5.9, "spät", 0.9)),
        _segment(0.0, 1.0, Word(0.1, 0.9, "früh", 0.9)),
    ]

    segments, _ = transcript_to_dataframes(transcript)

    assert list(segments.columns) == SEGMENT_COLUMNS
    assert segments["text"].tolist() == ["früh", "spät"]
    assert segments["segment_id"].tolist() == [0, 1]


def test_words_carry_the_segment_they_came_from_and_their_place_in_it():
    transcript = [
        _segment(
            0.0,
            3.0,
            Word(0.1, 0.5, "eins", 0.9),
            Word(1.0, 1.5, "zwei", 0.8),
        ),
        _segment(4.0, 5.0, Word(4.1, 4.9, "drei", 0.7)),
    ]

    _, words = transcript_to_dataframes(transcript)

    assert list(words.columns) == WORD_COLUMNS
    assert words["word"].tolist() == ["eins", "zwei", "drei"]
    assert words["segment_id"].tolist() == [0, 0, 1]
    assert words["word_index"].tolist() == [0, 1, 0]
    assert words["score"].tolist() == [0.9, 0.8, 0.7]


def test_words_are_stripped_of_whisper_padding():
    transcript = [_segment(0.0, 1.0, Word(0.1, 0.9, " hallo", 0.9))]

    _, words = transcript_to_dataframes(transcript)

    assert words["word"].tolist() == ["hallo"]


def test_an_empty_transcript_gives_empty_tables_with_the_right_columns():
    segments, words = transcript_to_dataframes([])

    assert list(segments.columns) == SEGMENT_COLUMNS
    assert list(words.columns) == WORD_COLUMNS
    assert segments.empty and words.empty
    assert segments["segment_id"].dtype == int
    assert words["word_index"].dtype == int


def test_a_segment_without_words_still_appears():
    transcript = [_segment(0.0, 1.0)]

    segments, words = transcript_to_dataframes(transcript)

    assert len(segments) == 1
    assert words.empty


def test_transcribe_reads_the_recording(data_dir):
    segments = list(
        transcribe(
            data_dir / "three-people-conversation.opus",
            model_name=CI_MODEL,
            language="de",
            device="cpu",
        )
    )

    assert segments
    assert all(s.end > s.start for s in segments)
    assert all(s.words for s in segments)
    text = " ".join(s.text for s in segments).lower()
    # The phrases even the tiny model gets right on this fixture.
    assert "guten morgen" in text
    assert "kaffee" in text
    assert "minuten" in text


def test_transcribe_runs_on_the_device_it_was_given(monkeypatch, data_dir):
    devices = []

    class _Model:
        def __init__(self, model_name, device, compute_type, download_root):
            devices.append(device)

        def transcribe(self, audio_path, **options):
            return [], None

    monkeypatch.setattr("faster_whisper.WhisperModel", _Model)

    list(
        transcribe(
            data_dir / "three-people-conversation.opus",
            model_name=CI_MODEL,
            device="cuda",
        )
    )

    assert devices == ["cuda"]


def test_transformers_checkpoint_is_converted_once(monkeypatch, tmp_path):
    import ctranslate2.converters
    import huggingface_hub

    conversions = []
    downloads = []
    tokenizer_downloads = []

    def _download(**options):
        downloads.append(options)
        source = tmp_path / "source"
        source.mkdir(exist_ok=True)
        return str(source)

    def _download_tokenizer(**options):
        tokenizer_downloads.append(options)
        tokenizer = tmp_path / "canonical-tokenizer.json"
        tokenizer.write_text("canonical")
        return str(tokenizer)

    class _Converter:
        def __init__(self, model_name, **options):
            conversions.append((model_name, options))

        def convert(self, output_dir, **options):
            conversions[-1] += (options,)
            output = Path(output_dir)
            output.mkdir()
            (output / "model.bin").write_bytes(b"converted")

    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", _Converter)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download_tokenizer)
    monkeypatch.setattr(
        "body_eye_sync.pipeline.transcription.model_cache_dir", lambda: tmp_path
    )

    from body_eye_sync.pipeline.transcription import _faster_whisper_model

    model_name = "primeline/whisper-large-v3-turbo-german"
    first = _faster_whisper_model(model_name)
    second = _faster_whisper_model(model_name)

    assert first == second
    assert Path(first, "model.bin").read_bytes() == b"converted"
    assert Path(first, "tokenizer.json").read_text() == "canonical"
    assert not list(tmp_path.glob("ctranslate2/.convert-*"))
    assert downloads == [
        {
            "repo_id": model_name,
            "cache_dir": tmp_path / "transformers",
            "allow_patterns": ["*.json", "*.safetensors", "*.txt"],
        }
    ]
    assert tokenizer_downloads == [
        {
            "repo_id": "Systran/faster-whisper-large-v3",
            "filename": "tokenizer.json",
            "cache_dir": tmp_path / "transformers",
        }
    ]
    assert conversions == [
        (
            str(tmp_path / "source"),
            {
                "copy_files": ["preprocessor_config.json"],
                "load_as_float16": True,
                "low_cpu_mem_usage": True,
            },
            {"quantization": "float16"},
        )
    ]


def test_native_faster_whisper_model_is_not_converted(monkeypatch):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.transcription._converted_model_path",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected conversion")),
    )

    from body_eye_sync.pipeline.transcription import _faster_whisper_model

    assert _faster_whisper_model("large-v3") == "large-v3"
