from body_eye_sync.pipeline.model_cache import cached_model_path


def test_cached_model_path_routes_bare_names_into_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.model_cache.user_cache_path",
        lambda appname, appauthor: tmp_path / appauthor / appname,
    )
    models_dir = tmp_path / "SSC" / "body-eye-sync" / "models"

    # Any bare weights name (not just one special-cased default) goes to the cache.
    assert cached_model_path("yolo26m.pt") == str(models_dir / "yolo26m.pt")
    assert cached_model_path("custom-pose.pt") == str(models_dir / "custom-pose.pt")
    assert models_dir.is_dir()


def test_cached_model_path_leaves_explicit_paths_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "body_eye_sync.pipeline.model_cache.user_cache_path",
        lambda appname, appauthor: tmp_path / appauthor / appname,
    )

    # A path with a directory part is used as given.
    explicit = tmp_path / "weights" / "custom-pose.pt"
    assert cached_model_path(explicit) == str(explicit)

    # An existing file in the cwd is not rewritten.
    here = tmp_path / "local.pt"
    here.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert cached_model_path("local.pt") == "local.pt"
