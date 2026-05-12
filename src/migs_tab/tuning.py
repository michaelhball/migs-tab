"""Detect the guitar tuning (and capo position) for a given video.

We try two strategies in order:

1. **Caption-driven.** Tutorial-style videos almost always have the instructor
   explicitly call out the tuning ("we're in drop D", "capo on the 2nd fret",
   "tune your low E down to D", "half-step down"). The Claude skill can do a
   richer job of this but we also do a pragmatic keyword scan here so the
   CLI can run standalone.

2. **Audio-driven.** Live recordings won't have captions saying the tuning,
   so we run librosa.pyin on the (preferably guitar-isolated) audio, find
   the lowest sustained pitch the player actually hits, and match it against
   a small library of common tuning candidates. We also check whether a
   capo would explain a chord set that doesn't fit standard.

Output is ``cache/<video_id>/tuning.json``:

```json
{
  "strings_midi": [40, 45, 50, 55, 59, 64],
  "capo": 0,
  "label": "Standard",
  "confidence": 0.9,
  "source": "audio" | "captions" | "default",
  "evidence": "..."
}
```
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .paths import VideoPaths

# ---------------------------------------------------------------------------
# Tuning library
# ---------------------------------------------------------------------------

# Each candidate tuning: low-to-high MIDI pitches of the open strings.
# Standard tuning low E is MIDI 40 (E2); add a constant to every string to
# shift up (capo) or down (whole-step etc.).
STANDARD_TUNING: list[int] = [40, 45, 50, 55, 59, 64]  # E A D G B E
DEFAULT_TUNING_LABEL = "Standard"

_TUNING_LIBRARY: dict[str, list[int]] = {
    "Standard": [40, 45, 50, 55, 59, 64],
    "Drop D": [38, 45, 50, 55, 59, 64],
    "Double Drop D": [38, 45, 50, 55, 59, 62],  # both E strings tuned to D
    "Half-step down": [39, 44, 49, 54, 58, 63],
    "Whole-step down": [38, 43, 48, 53, 57, 62],
    "Drop D, half-step down": [37, 44, 49, 54, 58, 63],
    "DADGAD": [38, 45, 50, 55, 57, 62],
    "Open D": [38, 45, 50, 54, 57, 62],
    "Open G": [38, 43, 50, 55, 59, 62],
    "Open E": [40, 47, 52, 56, 59, 64],
}


@dataclass
class Tuning:
    strings_midi: list[int]
    capo: int
    label: str
    confidence: float
    source: str
    evidence: str

    def effective_open_pitches(self) -> list[int]:
        """Open-string pitches AFTER the capo. A capo at fret N raises every
        open-string pitch by N semitones."""
        return [p + self.capo for p in self.strings_midi]

    def to_dict(self) -> dict:
        return {
            "strings_midi": self.strings_midi,
            "capo": self.capo,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "evidence": self.evidence,
        }


def default_tuning(reason: str = "no detection attempted") -> Tuning:
    return Tuning(
        strings_midi=list(STANDARD_TUNING),
        capo=0,
        label=DEFAULT_TUNING_LABEL,
        confidence=0.5,
        source="default",
        evidence=reason,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_tuning(paths: VideoPaths, force: bool = False) -> VideoPaths:
    """Detect and write ``tuning.json``. Idempotent unless ``force``."""
    if paths.tuning_json.exists() and not force:
        return paths

    # Strategy 1: captions, if present and non-empty.
    caption_t = _detect_from_captions(paths) if paths.captions_text.exists() else None
    if caption_t is not None and caption_t.confidence >= 0.8:
        paths.tuning_json.write_text(json.dumps(caption_t.to_dict(), indent=2))
        return paths

    # Strategy 2: audio. Prefer the isolated guitar stem; fall back to mixed.
    audio_source = paths.guitar_stem if paths.guitar_stem.exists() else paths.audio
    audio_t = _detect_from_audio(audio_source) if audio_source.exists() else None

    # Combine evidence — captions weak but present + audio strong = audio wins.
    chosen = audio_t or caption_t or default_tuning("no audio or captions usable")
    paths.tuning_json.write_text(json.dumps(chosen.to_dict(), indent=2))
    return paths


# ---------------------------------------------------------------------------
# Caption-driven detection
# ---------------------------------------------------------------------------

# Caption patterns intentionally require an explicit *tuning context* to fire.
# Guitarists say "open D string" / "open G note" / "drop the D" all the time
# without meaning the tuning — we only treat these as evidence when they
# appear next to words like "tuning", "tuned to", "we're in", or are a
# stand-alone-unambiguous keyword (DADGAD).
_CAPTION_PATTERNS: list[tuple[str, str, dict]] = [
    # DADGAD is unambiguous on its own.
    (r"\bdadgad\b", "DADGAD", {"strings_midi": _TUNING_LIBRARY["DADGAD"]}),
    # Double Drop D — must come BEFORE the bare Drop D pattern so it wins.
    # Accepts "double drop d tuning" / "tuned to double drop d" / etc.
    (
        r"\b(?:(?:tuned\s+to|tune\s+to|in|tuning(?:\s+is)?|we'?re\s+in|playing\s+in)\s+)?double\s+drop\s*d(?:\s+tuning)?\b",
        "Double Drop D",
        {"strings_midi": _TUNING_LIBRARY["Double Drop D"]},
    ),
    # Drop D: require tuning context (before or after).
    (
        r"\b(?:tuned\s+to|tune\s+to|tuning(?:\s+is)?|we'?re\s+in|playing\s+in)\s+drop\s*d\b",
        "Drop D",
        {"strings_midi": _TUNING_LIBRARY["Drop D"]},
    ),
    (
        r"\bdrop\s*d\s+tuning\b",
        "Drop D",
        {"strings_midi": _TUNING_LIBRARY["Drop D"]},
    ),
    # Open tunings: require "tuning" suffix.
    (
        r"\bopen\s*d(?:\s+tuning)\b",
        "Open D",
        {"strings_midi": _TUNING_LIBRARY["Open D"]},
    ),
    (
        r"\bopen\s*g(?:\s+tuning)\b",
        "Open G",
        {"strings_midi": _TUNING_LIBRARY["Open G"]},
    ),
    (
        r"\bopen\s*e(?:\s+tuning)\b",
        "Open E",
        {"strings_midi": _TUNING_LIBRARY["Open E"]},
    ),
    # Half-step / whole-step down: these phrases ARE tuning-specific, no context needed.
    (
        r"\bhalf[-\s]*step[-\s]*(?:down|flat)\b",
        "Half-step down",
        {"strings_midi": _TUNING_LIBRARY["Half-step down"]},
    ),
    (
        r"\b(?:whole[-\s]*step|one[-\s]*step)[-\s]*(?:down|flat)\b",
        "Whole-step down",
        {"strings_midi": _TUNING_LIBRARY["Whole-step down"]},
    ),
    # "Tune your low E down to D" → Drop D.
    (
        r"\btune\s+(?:your\s+)?(?:low\s+)?e\s+(?:down\s+)?to\s+d\b",
        "Drop D",
        {"strings_midi": _TUNING_LIBRARY["Drop D"]},
    ),
]

_CAPO_PATTERN = re.compile(
    r"capo\s*(?:on|at)?\s*(?:the\s*)?(\d{1,2})(?:st|nd|rd|th)?\s*(?:fret)?",
    re.IGNORECASE,
)


def _detect_from_captions(paths: VideoPaths) -> Tuning | None:
    text = paths.captions_text.read_text(encoding="utf-8", errors="ignore").lower()
    if not text.strip():
        return None

    capo = 0
    capo_match = _CAPO_PATTERN.search(text)
    if capo_match:
        try:
            capo = int(capo_match.group(1))
            if not 0 <= capo <= 12:
                capo = 0
        except ValueError:
            capo = 0

    for pattern, label, kwargs in _CAPTION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return Tuning(
                strings_midi=list(kwargs["strings_midi"]),
                capo=capo,
                label=label + (f", capo {capo}" if capo else ""),
                confidence=0.9,
                source="captions",
                evidence=f"matched '{m.group(0)}' in captions",
            )

    if capo > 0:
        # No tuning keyword but explicit capo — assume standard.
        return Tuning(
            strings_midi=list(STANDARD_TUNING),
            capo=capo,
            label=f"Standard, capo {capo}",
            confidence=0.85,
            source="captions",
            evidence=f"matched capo {capo} in captions",
        )

    return None


# ---------------------------------------------------------------------------
# Audio-driven detection
# ---------------------------------------------------------------------------


# Candidate (base_tuning_label, base_pitches, capo) pairs to test against the
# detected lowest sustained pitch. The capo extension is what gives us
# audio capo detection — the same Standard tuning is tried at every capo
# position from 0 to 7, and the closest match wins.
_AUDIO_CANDIDATES: list[tuple[str, list[int], int]] = []


def _build_audio_candidates() -> list[tuple[str, list[int], int]]:
    capo_eligible = {
        "Standard": list(range(0, 8)),  # 0..7
        "Drop D": list(range(0, 4)),
        "Half-step down": [0],
        "Whole-step down": [0],
        "Drop D, half-step down": [0],
        "DADGAD": [0],
        "Open D": [0],
        "Open G": [0],
        "Open E": [0],
    }
    out: list[tuple[str, list[int], int]] = []
    for label, midis in _TUNING_LIBRARY.items():
        capos = capo_eligible.get(label, [0])
        for c in capos:
            out.append((label, midis, c))
    return out


_AUDIO_CANDIDATES = _build_audio_candidates()


def _detect_from_audio(audio_path: Path) -> Tuning | None:
    """Heuristic: load up to ~120s of audio, run pyin to get sustained pitches,
    find the lowest reliably-detected MIDI pitch, and match to a
    (tuning, capo) pair from the candidate library."""
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=120.0)
    except Exception as e:
        return Tuning(
            strings_midi=list(STANDARD_TUNING),
            capo=0,
            label=DEFAULT_TUNING_LABEL,
            confidence=0.3,
            source="default",
            evidence=f"audio load failed: {e!r}",
        )

    if y.size == 0:
        return None

    fmin = float(librosa.midi_to_hz(34))  # below all common tunings (low B = 35)
    fmax = float(librosa.midi_to_hz(70))  # higher than any open-string fundamental
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, sr=sr, fmin=fmin, fmax=fmax, frame_length=2048
        )
    except Exception as e:
        return Tuning(
            strings_midi=list(STANDARD_TUNING),
            capo=0,
            label=DEFAULT_TUNING_LABEL,
            confidence=0.3,
            source="default",
            evidence=f"pyin failed: {e!r}",
        )

    valid_mask = (~np.isnan(f0)) & (voiced_probs > 0.6)
    valid_f0 = f0[valid_mask]
    if valid_f0.size < 20:
        return None

    # Lowest reliably-sustained frequency (2nd percentile, robust to outliers).
    low_hz = float(np.percentile(valid_f0, 2))
    low_midi = float(librosa.hz_to_midi(low_hz))

    # Score every (tuning, capo) candidate by how close its effective low-E
    # pitch is to the detected lowest pitch. Tie-break: prefer no capo, then
    # smaller capo numbers (capo'd songs are still less common than open).
    def score(c: tuple[str, list[int], int]) -> tuple[float, int]:
        _, midis, capo = c
        return abs(low_midi - (midis[0] + capo)), capo

    candidates = sorted(_AUDIO_CANDIDATES, key=score)
    best_label, best_midis, best_capo = candidates[0]
    effective_low = best_midis[0] + best_capo
    delta = low_midi - effective_low

    # Confidence based on how close the detected pitch is to the candidate.
    confidence = max(0.0, 1.0 - min(1.0, abs(delta) / 1.0))

    final_label = best_label if best_capo == 0 else f"{best_label}, capo {best_capo}"

    return Tuning(
        strings_midi=list(best_midis),
        capo=best_capo,
        label=final_label,
        confidence=round(confidence, 3),
        source="audio",
        evidence=(
            f"lowest sustained pitch ≈ MIDI {low_midi:.2f} "
            f"({librosa.midi_to_note(low_midi, octave=True)}); "
            f"closest match: {final_label} "
            f"(effective low {effective_low}, offset {delta:+.2f} st)"
        ),
    )
