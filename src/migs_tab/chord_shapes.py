"""Phase 3.5 (chord-shape track): pick one representative video frame per
distinct chord in the song's progression, so the Claude skill can vision-check
the actual finger placement and confirm or correct the algorithm's chord
templates.

This is cheaper than the existing per-note vision pass — one frame per
unique chord name = ~6-15 frames per song, regardless of the song's length.
Frames are saved to ``cache/<id>/frames/chord-shapes/<chord>.jpg`` and a
``chord-shape-candidates.json`` records each chord's representative time
plus all available spans (so the skill can grab a different frame if the
first one shows the hand mid-transition).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import frames as frames_mod
from .paths import VideoPaths

# Minimum span duration to count as a "verifiable" chord — shorter spans
# usually catch passing chords or basic-pitch flicker and the hand might
# not be fully settled.
_MIN_SPAN_DURATION = 1.0

# How far into the chord span to sample the frame. 0.25 = past the initial
# strum (when fingers are still moving) but before any chord-change motion
# toward the end.
_FRAME_SAMPLE_RATIO = 0.25

# Cap the number of candidate frames per chord. The skill grabs the first
# one as the primary; the rest are alternates if the primary is blurry /
# mid-transition.
_MAX_CANDIDATES_PER_CHORD = 4


def select_and_extract(paths: VideoPaths, force: bool = False) -> Path:
    """Pick representative frames for each unique chord; write a candidates
    JSON listing the chord, timestamp, and frame path.

    Returns the path to ``chord-shape-candidates.json``.
    """
    if paths.chord_shape_candidates_json.exists() and not force:
        return paths.chord_shape_candidates_json
    if not paths.structure_json.exists():
        raise FileNotFoundError(
            f"structure.json not found at {paths.structure_json}; run `migs-tab structure` first."
        )
    if not paths.video.exists():
        raise FileNotFoundError(
            f"video.mp4 not found at {paths.video}; run `migs-tab download` first."
        )

    data = json.loads(paths.structure_json.read_text())

    # Collect all chord spans across all playing segments.
    spans_by_chord: dict[str, list[dict]] = defaultdict(list)
    for seg in data.get("playing_segments", []):
        for c in seg.get("chords", []):
            duration = c["end"] - c["start"]
            if duration < _MIN_SPAN_DURATION:
                continue
            spans_by_chord[c["chord"]].append(
                {
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "duration": float(duration),
                }
            )

    out_dir = paths.frames_dir / "chord-shapes"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, dict] = {}
    for chord, spans in spans_by_chord.items():
        # Sort by duration, longest first — longer spans give the player more
        # time to settle into a stable fingering.
        spans.sort(key=lambda s: s["duration"], reverse=True)
        primary = spans[0]
        primary_ts = primary["start"] + primary["duration"] * _FRAME_SAMPLE_RATIO
        primary_frame = frames_mod.extract_frame(
            paths, primary_ts, out_dir=out_dir, label=_safe_label(chord)
        )

        alternates: list[dict] = []
        for span in spans[1:_MAX_CANDIDATES_PER_CHORD]:
            ts = span["start"] + span["duration"] * _FRAME_SAMPLE_RATIO
            alternates.append(
                {
                    "timestamp": round(ts, 3),
                    "span_start": round(span["start"], 3),
                    "span_end": round(span["end"], 3),
                    "duration": round(span["duration"], 3),
                }
            )

        candidates[chord] = {
            "primary_timestamp": round(primary_ts, 3),
            "primary_frame": str(primary_frame),
            "primary_span_start": round(primary["start"], 3),
            "primary_span_end": round(primary["end"], 3),
            "primary_duration": round(primary["duration"], 3),
            "total_instances": len(spans),
            "alternates": alternates,
        }

    out = {
        "video_id": data.get("video_id"),
        "chord_count": len(candidates),
        "candidates": candidates,
    }
    paths.chord_shape_candidates_json.write_text(json.dumps(out, indent=2))
    return paths.chord_shape_candidates_json


def _safe_label(chord: str) -> str:
    """Sanitize a chord name for use in a filename."""
    return chord.replace("#", "sharp").replace("/", "_over_")
