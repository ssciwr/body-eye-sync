#!/usr/bin/env bash
# Fit and plot the quartet archive with landmark alignment and clock correction.
#
# The GoPro files are consecutive chunks of one recording. Their audio is
# concatenated before comparison with glasses recordings 1401--1404.
#
# Usage: scripts/plot_session_alignment.sh [output.png]
#
# Set ARCHIVE to use a different archive root. WINDOW, STEP and SEARCH override
# the local spectral-alignment settings.

set -euo pipefail

ARCHIVE="${ARCHIVE:-/data/body-eye-sync/Archiv}"
WINDOW="${WINDOW:-10}"
STEP="${STEP:-5}"
SEARCH="${SEARCH:-12}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$(realpath -m "${1:-$REPO/scripts/session_alignment_shifts.png}")"
GOPRO="$ARCHIVE/GoPro files"
GLASSES="$ARCHIVE/Video and audio files/Glasses videos plus synchronized gaze data"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
for directory in "$GOPRO" "$GLASSES"; do
    [ -d "$directory" ] || { echo "not found: $directory (set ARCHIVE?)" >&2; exit 1; }
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "concatenating GoPro audio..."
: > "$work/list.txt"
for chunk in "$GOPRO"/*.MP4; do
    printf "file '%s'\n" "$chunk" >> "$work/list.txt"
done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$work/list.txt" \
    -vn -ac 1 -ar 16000 "$work/gopro.wav"

cd "$REPO"
uv run python scripts/plot_alignment_timeline.py "$OUTPUT" \
    "gopro=$work/gopro.wav" \
    "1401=$GLASSES/G3 1401.mp4" \
    "1402=$GLASSES/G3 1402.mp4" \
    "1403=$GLASSES/G2 1403.mp4" \
    "1404=$GLASSES/G2 1404-02.mp4" \
    --reference gopro --window "$WINDOW" --step "$STEP" --search "$SEARCH"
