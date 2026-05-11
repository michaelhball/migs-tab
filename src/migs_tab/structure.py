"""Phase 2: structural analysis of a guitar tutorial.

For each video we produce ``structure.json`` containing:
  - The audio duration and sample-level metadata.
  - A list of *playing segments* — contiguous regions where the instructor is
    actively playing the guitar (as opposed to talking or silence).
  - For each playing segment: average loudness, a coarse chord progression
    derived from chroma + template matching, and the captions said during
    (or just before) the segment.

The LLM step (executed by the Claude skill in-session) consumes this file
along with the raw captions and produces ``sections.json`` mapping each
playing segment to its role in the song (intro / verse / chorus / etc.)
and grouping repetitions of the same role together.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from .paths import VideoPaths

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Sample rate for analysis (downsampled for speed; chroma is robust at 22050).
_SR = 22050
# librosa.effects.split top_db. Higher = more permissive (treats more as
# non-silent). 30 dB is a good starting point for a Demucs guitar stem;
# bleed/artifacts sit around -40 dB.
_TOP_DB = 30
# Minimum playing-segment length (s). Shorter blips get discarded.
_MIN_SEGMENT_DURATION = 1.5
# Gap to merge across between consecutive active regions (s). A short pause
# while the instructor breathes / lets a chord ring shouldn't break a segment.
_MERGE_GAP = 0.8
# Chroma analysis hop (in samples at _SR). 0.5s windows ~ enough to resolve
# typical chord changes in a tutorial without overfitting to single beats.
_CHROMA_HOP_SECONDS = 0.25
# Minimum chord duration when collapsing the per-frame chord sequence.
_MIN_CHORD_DURATION = 0.5
# Caption context window: include captions that ended within this many
# seconds before a segment, since the instructor usually announces then plays.
_CAPTION_LOOKBACK = 5.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class Caption:
    start: float
    end: float
    text: str


@dataclass
class ChordSpan:
    chord: str
    start: float
    end: float


@dataclass
class PlayingSegment:
    id: int
    start: float
    end: float
    duration: float
    rms_mean: float
    chords: list[ChordSpan]
    captions: list[Caption]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "rms_mean": round(self.rms_mean, 4),
            "chords": [
                {
                    "chord": c.chord,
                    "start": round(c.start, 3),
                    "end": round(c.end, 3),
                }
                for c in self.chords
            ],
            "captions": [
                {
                    "start": round(c.start, 2),
                    "end": round(c.end, 2),
                    "text": c.text,
                }
                for c in self.captions
            ],
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_structure(paths: VideoPaths, force: bool = False) -> VideoPaths:
    """Compute structure.json for a video. Idempotent unless ``force``."""
    if paths.structure_json.exists() and not force:
        return paths

    if not paths.guitar_stem.exists():
        raise FileNotFoundError(
            f"Guitar stem not found at {paths.guitar_stem}. Run separate first."
        )

    y, sr = librosa.load(str(paths.guitar_stem), sr=_SR, mono=True)
    total_duration = float(len(y) / sr)

    raw_segments = _detect_playing_intervals(y, sr)
    captions = (
        _parse_vtt(paths.captions_vtt) if paths.captions_vtt.exists() else []
    )

    templates = _build_chord_templates()

    segments: list[PlayingSegment] = []
    for idx, (seg_start, seg_end) in enumerate(raw_segments):
        seg_audio = y[int(seg_start * sr) : int(seg_end * sr)]
        rms_mean = float(np.sqrt(np.mean(seg_audio.astype(np.float64) ** 2)))
        chords = _detect_chord_progression(seg_audio, sr, templates, seg_start)
        seg_captions = _captions_for_segment(captions, seg_start, seg_end)
        segments.append(
            PlayingSegment(
                id=idx,
                start=seg_start,
                end=seg_end,
                duration=seg_end - seg_start,
                rms_mean=rms_mean,
                chords=chords,
                captions=seg_captions,
            )
        )

    out = {
        "video_id": paths.video_id,
        "audio_path": str(paths.guitar_stem.relative_to(paths.root.parent.parent)),
        "audio_duration": round(total_duration, 3),
        "sample_rate": sr,
        "params": {
            "top_db": _TOP_DB,
            "min_segment_duration_s": _MIN_SEGMENT_DURATION,
            "merge_gap_s": _MERGE_GAP,
            "chroma_hop_s": _CHROMA_HOP_SECONDS,
            "min_chord_duration_s": _MIN_CHORD_DURATION,
        },
        "playing_segment_count": len(segments),
        "playing_segments": [s.to_dict() for s in segments],
    }
    paths.structure_json.write_text(json.dumps(out, indent=2))
    return paths


# ---------------------------------------------------------------------------
# Playing-segment detection
# ---------------------------------------------------------------------------


def _detect_playing_intervals(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Return (start_s, end_s) for each region of sustained playing.

    Strategy: librosa.effects.split for initial non-silent regions, then
    merge close-together regions and drop too-short ones.
    """
    intervals = librosa.effects.split(y, top_db=_TOP_DB)
    spans = [(int(a) / sr, int(b) / sr) for a, b in intervals]

    # Merge near-adjacent spans.
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= _MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Drop too-short spans (likely transients / talking with brief twangs).
    return [
        (s, e) for s, e in merged if (e - s) >= _MIN_SEGMENT_DURATION
    ]


# ---------------------------------------------------------------------------
# Chord recognition
# ---------------------------------------------------------------------------


def _build_chord_templates() -> dict[str, np.ndarray]:
    """{chord_name: normalized 12-dim pitch-class vector} for maj, min, 7."""
    templates: dict[str, np.ndarray] = {}
    intervals = {
        "": (0, 4, 7),  # major
        "m": (0, 3, 7),  # minor
        "7": (0, 4, 7, 10),  # dominant seventh
    }
    for root in range(12):
        for suffix, ivals in intervals.items():
            vec = np.zeros(12, dtype=np.float64)
            for i in ivals:
                vec[(root + i) % 12] = 1.0
            templates[f"{PITCH_NAMES[root]}{suffix}"] = vec / np.linalg.norm(vec)
    return templates


def _detect_chord_progression(
    seg_audio: np.ndarray,
    sr: int,
    templates: dict[str, np.ndarray],
    seg_start_offset: float,
) -> list[ChordSpan]:
    """Match chroma frames against chord templates, smooth, collapse to spans."""
    if len(seg_audio) < int(sr * _CHROMA_HOP_SECONDS):
        return []

    hop_length = int(sr * _CHROMA_HOP_SECONDS)
    # CQT chroma is more pitch-stable than STFT chroma — better for guitar
    # which has strong harmonics that smear STFT bins.
    chroma = librosa.feature.chroma_cqt(
        y=seg_audio,
        sr=sr,
        hop_length=hop_length,
        n_chroma=12,
    )

    # Normalize each column.
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    chroma = chroma / norms

    # Score each frame against every template.
    chord_names = list(templates.keys())
    template_matrix = np.stack([templates[n] for n in chord_names])  # (C, 12)
    scores = template_matrix @ chroma  # (C, T)
    best_idx = np.argmax(scores, axis=0)
    best_chord_per_frame = [chord_names[i] for i in best_idx]

    times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)

    spans: list[ChordSpan] = []
    current_chord = best_chord_per_frame[0]
    current_start = 0.0
    for i in range(1, len(best_chord_per_frame)):
        if best_chord_per_frame[i] != current_chord:
            spans.append(
                ChordSpan(
                    chord=current_chord,
                    start=seg_start_offset + current_start,
                    end=seg_start_offset + float(times[i]),
                )
            )
            current_chord = best_chord_per_frame[i]
            current_start = float(times[i])
    # final span
    spans.append(
        ChordSpan(
            chord=current_chord,
            start=seg_start_offset + current_start,
            end=seg_start_offset + float(times[-1]) + _CHROMA_HOP_SECONDS,
        )
    )

    return _smooth_chord_spans(spans)


def _smooth_chord_spans(spans: list[ChordSpan]) -> list[ChordSpan]:
    """Drop chord spans shorter than _MIN_CHORD_DURATION by merging into
    whichever neighbor is louder/longer. Then collapse consecutive identical
    chords (which appear after merging)."""
    if not spans:
        return spans

    # Pass 1: drop tiny spans, attaching their time to whichever neighbor is longer.
    while True:
        too_short = [
            i for i, s in enumerate(spans) if (s.end - s.start) < _MIN_CHORD_DURATION
        ]
        if not too_short:
            break
        i = too_short[0]
        if i == 0 and len(spans) > 1:
            spans[1] = ChordSpan(spans[1].chord, spans[0].start, spans[1].end)
            spans.pop(0)
        elif i == len(spans) - 1 and len(spans) > 1:
            spans[-2] = ChordSpan(spans[-2].chord, spans[-2].start, spans[-1].end)
            spans.pop()
        elif len(spans) == 1:
            break
        else:
            left = spans[i - 1]
            right = spans[i + 1]
            if (left.end - left.start) >= (right.end - right.start):
                spans[i - 1] = ChordSpan(left.chord, left.start, spans[i].end)
            else:
                spans[i + 1] = ChordSpan(right.chord, spans[i].start, right.end)
            spans.pop(i)

    # Pass 2: collapse consecutive identical chords.
    collapsed: list[ChordSpan] = []
    for s in spans:
        if collapsed and collapsed[-1].chord == s.chord:
            collapsed[-1] = ChordSpan(s.chord, collapsed[-1].start, s.end)
        else:
            collapsed.append(s)
    return collapsed


# ---------------------------------------------------------------------------
# VTT parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _parse_vtt(vtt_path: Path) -> list[Caption]:
    """Parse a YouTube auto-caption VTT into a deduplicated Caption list.

    YouTube's auto-captions emit overlapping cues — each cue extends the
    previous with one or two new words. We dedupe by collapsing each cue
    down to its final visible line (the most-complete version).
    """
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    captions: list[Caption] = []

    current_time: tuple[float, float] | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_time is None or not current_lines:
            return
        # Take the *last* non-empty line, which is the fully accumulated text
        # for auto-captions, then strip tags.
        for raw in reversed(current_lines):
            clean = _TAG_RE.sub("", raw).strip()
            if clean:
                captions.append(Caption(current_time[0], current_time[1], clean))
                return

    for raw in text.splitlines():
        line = raw.rstrip()
        m = _TIMESTAMP_RE.search(line)
        if m:
            _flush()
            sh, sm, ss, sms, eh, em, es, ems = map(int, m.groups())
            start = sh * 3600 + sm * 60 + ss + sms / 1000.0
            end = eh * 3600 + em * 60 + es + ems / 1000.0
            current_time = (start, end)
            current_lines = []
            continue
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        current_lines.append(line)
    _flush()

    # Dedupe — auto-captions repeat the same final text many times across
    # overlapping cues. Keep only the first occurrence of each text.
    seen: set[str] = set()
    deduped: list[Caption] = []
    for c in captions:
        key = c.text
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _captions_for_segment(
    captions: Iterable[Caption], seg_start: float, seg_end: float
) -> list[Caption]:
    """Return captions that overlap [seg_start - lookback, seg_end]."""
    window_start = seg_start - _CAPTION_LOOKBACK
    return [c for c in captions if c.end >= window_start and c.start <= seg_end]
