# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- add `GlassesVideo`, `FixedVideo` and `Audio` input types [#31](https://github.com/ssciwr/body-eye-sync/pull/31)
- split the GUI into a tab per stage, with an Input files tab for managing an experiment's inputs [#34](https://github.com/ssciwr/body-eye-sync/pull/34)
- add a speech pipeline: transcription [#35](https://github.com/ssciwr/body-eye-sync/pull/35)
- automatically align recordings, detect and fix clock drift and gaps [#36](https://github.com/ssciwr/body-eye-sync/pull/36)
- add synchronised video export functionality and GUI export tab [??](??)
- add an Audio processing tab to transcribe any input that carries audio
- show transcript segments in the Audio processing tab as Whisper produces them
- install the CUDA 12 build of torch, which is the CUDA version CTranslate2 needs to transcribe on a GPU
- add a `device` setting to choose where transcription runs
- default to the accuracy-tuned primeLine German Whisper model and convert its
  official checkpoint for faster-whisper on first use
- remove speaker diarization: speakers now come from whose microphone heard them loudest
- add a Speech post processing tab that works out an experiment's speech turns by comparing its glasses recordings against each other
- write speech turns and their word-level timestamps as ELAN annotations
  alongside the combined video, with turn and word tiers for each speaker

## [0.0.4] - 2026-07-03

- First PyPI pre-release.
