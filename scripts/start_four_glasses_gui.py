from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# AI-generated, just to help with testing the alignment tab and general GUI without spending time on selecting files over and over.

ROOT = Path(__file__).resolve().parents[1]
GLASSES_DIR = (
    ROOT
    / "data"
    / "Video and audio files"
    / "Glasses videos plus synchronized gaze data"
)
SPLITSCREEN_DIR = ROOT / "data" / "Video and audio files" / "Splitscreen file"
EXPERIMENT_DIR = Path(tempfile.gettempdir()) / "body-eye-sync-four-glasses"
VIDEOS = (
    ("g3-1401", "G3 1401.mp4", "G3_1401mp4.tsv"),
    ("g3-1402", "G3 1402.mp4", "G3_1402mp4.tsv"),
    ("g2-1403", "G2 1403.mp4", "G2_1403mp4.tsv"),
    ("g2-1404", "G2 1404-02.mp4", "G2_1404mp4_quartet.tsv"),
)
AUDIO = ("quartet-audio", "Gruppe 1400_Quartett_Audio.wav")

missing = [
    str(GLASSES_DIR / filename)
    for _input_id, video, gaze in VIDEOS
    for filename in (video, gaze)
    if not (GLASSES_DIR / filename).exists()
]
if not (SPLITSCREEN_DIR / AUDIO[1]).exists():
    missing.append(str(SPLITSCREEN_DIR / AUDIO[1]))
if missing:
    raise SystemExit("Missing sample files:\n" + "\n".join(missing))

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
(EXPERIMENT_DIR / "experiment.yaml").write_text(
    yaml.safe_dump(
        {
            "version": 1,
            "glasses_videos": [
                {
                    "id": input_id,
                    "path": str(GLASSES_DIR / video),
                    "gaze_path": str(GLASSES_DIR / gaze),
                }
                for input_id, video, gaze in VIDEOS
            ],
            "audio": [
                {
                    "id": AUDIO[0],
                    "path": str(SPLITSCREEN_DIR / AUDIO[1]),
                }
            ],
        },
        sort_keys=False,
    ),
    encoding="utf-8",
)

subprocess.run([sys.executable, "-m", "body_eye_sync", str(EXPERIMENT_DIR)], check=True)
