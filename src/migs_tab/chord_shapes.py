"""Phase 3.5 (chord-shape track): pick the best video frame per distinct
chord so the Claude skill can vision-check the player's actual fingering.

The "best" frame for each chord is the result of two filters:

1. **Prefer earlier occurrences.** In tutorial videos the chord is usually
   introduced at a cowboy / first-position fingering early on, then later
   appears in alternate voicings (up the neck, partial barres, etc.). The
   earliest reasonably-long span is the most likely to catch the canonical
   shape.

2. **Pick the sharpest frame within the chosen span(s).** We extract several
   candidate frames evenly spaced inside each of the first few qualifying
   spans, score each by Laplacian variance (a classic blurriness measure),
   and keep the sharpest one. This is a cheap, deterministic alternative
   to letting the LLM pick.

Each chord's representative frame ends up at
``cache/<id>/frames/chord-shapes/<chord>.jpg``. The candidates JSON
records the chosen frame's timestamp, sharpness score, and a list of
runner-up alternates the skill can fall back to.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from . import frames as frames_mod
from .paths import VideoPaths

# --- Span selection ----------------------------------------------------------

# Minimum span duration to count as a "verifiable" chord — the hand is
# usually settled into the fingering only after a beat or so.
_MIN_SPAN_DURATION = 1.0

# Sample frames at these fractions within a single span. Avoiding the very
# edges (0 = right before strum, 1 = right before next chord change) catches
# the middle where the hand is most stable.
_FRAME_SAMPLE_RATIOS = (0.30, 0.50, 0.70)

# Number of distinct span-times to sample per chord. We pick them evenly
# spaced through ALL of the chord's qualifying instances (sorted by
# midpoint time), so the resulting candidates are spread across the whole
# video rather than clumped at the start. This is critical because a
# tutorial often opens with a play-through demo that catches the hand at
# atypical voicings — sampling across the full song gives the LLM the
# best chance of seeing the canonical cowboy fingering.
_SPAN_SAMPLES_PER_CHORD = 6


# --- Image-quality scoring ---------------------------------------------------

# 3x3 Laplacian kernel. Variance of this filter's output correlates with
# image sharpness: blurry images have small variance, sharp ones large.
_LAPLACIAN_KERNEL = np.array(
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    dtype=np.float64,
)

# When scoring sharpness, downsample to this width for speed (~5x faster).
# Sharpness ranking is preserved at lower resolution.
_SHARPNESS_SCORE_WIDTH = 640


def _frame_sharpness(image_path: Path) -> float:
    """Higher = sharper. Variance of the Laplacian of the grayscale image."""
    img = Image.open(image_path).convert("L")
    if img.width > _SHARPNESS_SCORE_WIDTH:
        ratio = _SHARPNESS_SCORE_WIDTH / img.width
        img = img.resize((_SHARPNESS_SCORE_WIDTH, int(img.height * ratio)))
    arr = np.asarray(img, dtype=np.float64)
    lap = convolve2d(arr, _LAPLACIAN_KERNEL, mode="valid")
    return float(np.var(lap))


# --- Public entry point -------------------------------------------------------


def select_and_extract(paths: VideoPaths, force: bool = False) -> Path:
    """Pick the best representative frame per distinct chord. Returns the
    path to ``chord-shape-candidates.json``.

    Idempotent unless ``force`` — skips if the candidates JSON already
    exists.
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

    # Group spans by chord.
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
    tmp_dir = paths.frames_dir / "_chord-shapes-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, dict] = {}
    for chord, spans in spans_by_chord.items():
        spans.sort(key=lambda s: s["start"])  # by time
        sampled_spans = _evenly_sampled_spans(spans, _SPAN_SAMPLES_PER_CHORD)

        # For each sampled span, pick the sharpest frame within it. Keep
        # all of these as candidates so the skill has temporal diversity
        # (early vs mid vs late video) when choosing the canonical voicing.
        per_span_best: list[dict] = []
        for span_idx, span in enumerate(sampled_spans):
            in_span: list[dict] = []
            for ratio in _FRAME_SAMPLE_RATIOS:
                ts = span["start"] + span["duration"] * ratio
                frame_path = frames_mod.extract_frame(
                    paths,
                    ts,
                    out_dir=tmp_dir,
                    label=f"{_safe_label(chord)}_{span_idx}_{int(ratio * 100)}",
                    overwrite=True,
                )
                try:
                    sharpness = _frame_sharpness(frame_path)
                except Exception:
                    sharpness = 0.0
                in_span.append(
                    {
                        "timestamp": round(ts, 3),
                        "sharpness": round(sharpness, 1),
                        "span_start": round(span["start"], 3),
                        "span_end": round(span["end"], 3),
                        "span_duration": round(span["duration"], 3),
                        "_tmp_path": frame_path,
                    }
                )
            if not in_span:
                continue
            in_span.sort(key=lambda x: x["sharpness"], reverse=True)
            best_in_span = in_span[0]
            # Move this span's best frame to a stable name and drop the rest.
            final_path = out_dir / f"{_safe_label(chord)}_s{span_idx}.jpg"
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(best_in_span["_tmp_path"]), str(final_path))
            for entry in in_span[1:]:
                try:
                    entry["_tmp_path"].unlink()
                except FileNotFoundError:
                    pass
            best_in_span["frame_path"] = str(final_path)
            best_in_span.pop("_tmp_path", None)
            per_span_best.append(best_in_span)

        if not per_span_best:
            continue
        # Sort the per-span-bests by sharpness — the first is the "primary"
        # but the skill should still be able to use any of them.
        per_span_best.sort(key=lambda x: x["sharpness"], reverse=True)
        primary = per_span_best[0]

        candidates[chord] = {
            "primary_timestamp": primary["timestamp"],
            "primary_frame": primary["frame_path"],
            "primary_sharpness": primary["sharpness"],
            "primary_span_start": primary["span_start"],
            "primary_span_end": primary["span_end"],
            "primary_span_duration": primary["span_duration"],
            "total_instances": len(spans),
            # Every sampled span gets one entry so the skill can pick a
            # different time if the primary catches an atypical voicing.
            "candidates": [
                {
                    "timestamp": s["timestamp"],
                    "sharpness": s["sharpness"],
                    "span_start": s["span_start"],
                    "span_end": s["span_end"],
                    "frame_path": s["frame_path"],
                }
                for s in per_span_best
            ],
        }

    # Clean up the scratch directory (will be empty after the moves).
    try:
        tmp_dir.rmdir()
    except OSError:
        # leftover files (rare) — leave the dir
        pass

    out = {
        "video_id": data.get("video_id"),
        "chord_count": len(candidates),
        "params": {
            "min_span_duration_s": _MIN_SPAN_DURATION,
            "span_samples_per_chord": _SPAN_SAMPLES_PER_CHORD,
            "frames_sampled_per_span": len(_FRAME_SAMPLE_RATIOS),
            "sharpness_metric": "laplacian-variance",
        },
        "candidates": candidates,
    }
    paths.chord_shape_candidates_json.write_text(json.dumps(out, indent=2))
    return paths.chord_shape_candidates_json


def _evenly_sampled_spans(spans: list[dict], n: int) -> list[dict]:
    """Pick ``n`` spans evenly distributed across the (time-sorted) list.

    If the chord has fewer than ``n`` qualifying instances, return them
    all. Otherwise pick at evenly spaced indices so the resulting set
    spans the full range of times — critical for variety since tutorial
    demos and teaching segments use very different voicings.
    """
    if len(spans) <= n:
        return list(spans)
    indices = [int(round(i * (len(spans) - 1) / (n - 1))) for i in range(n)]
    # De-dupe (consecutive duplicates can occur when n is close to len(spans)).
    seen: set[int] = set()
    picked: list[dict] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            picked.append(spans[i])
    return picked


def _safe_label(chord: str) -> str:
    """Sanitize a chord name for use in a filename."""
    return chord.replace("#", "sharp").replace("/", "_over_")
