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

Whichever strategy wins, the candidate is then **cross-checked against the
transcription** (``notes.mt3.json``, falling back to ``notes.json``) when one
exists: sustained, repeated notes below the candidate's effective lowest
sounding pitch (lowest open string + capo) are physically impossible, so they
veto the candidate and we re-fit to the transcription's own pitch floor with
reduced confidence (see :func:`verify_against_transcription`). A symmetric
guard catches the opposite error: a DROPPED tuning (floor below MIDI 40)
whose low floor is only phantom sub-octave doublings of a real note an octave
up is rejected and the floor raised to the real one. If no transcription
exists yet, the result is kept but marked unverified and its confidence is
capped below 1.0.

Output is ``cache/<video_id>/tuning.json``:

```json
{
  "strings_midi": [40, 45, 50, 55, 59, 64],
  "capo": 0,
  "label": "Standard",
  "confidence": 0.9,
  "source": "audio" | "captions" | "default" | "<source>+contradiction-veto",
  "evidence": "...",
  "verification": {"checked": true, "subfloor_sustained_count": 0, ...}
}
```
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, replace
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
    # Result of the transcription cross-check (None until it has run).
    # See verify_against_transcription() for the keys recorded here.
    verification: dict | None = None

    def effective_open_pitches(self) -> list[int]:
        """Open-string pitches AFTER the capo. A capo at fret N raises every
        open-string pitch by N semitones."""
        return [p + self.capo for p in self.strings_midi]

    def to_dict(self) -> dict:
        d = {
            "strings_midi": self.strings_midi,
            "capo": self.capo,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "evidence": self.evidence,
        }
        if self.verification is not None:
            d["verification"] = self.verification
        return d


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
        chosen = caption_t
    else:
        # Strategy 2: audio. Prefer the isolated guitar stem; fall back to mixed.
        audio_source = paths.guitar_stem if paths.guitar_stem.exists() else paths.audio
        audio_t = _detect_from_audio(audio_source) if audio_source.exists() else None

        # Combine evidence — captions weak but present + audio strong = audio wins.
        chosen = audio_t or caption_t or default_tuning("no audio or captions usable")

    # Strategy 3: cross-check the winner against the transcription — sustained
    # notes below the candidate's effective floor veto it (see module docstring).
    chosen = verify_against_transcription(chosen, paths)
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
    primary_low_delta = abs(low_midi - (candidates[0][1][0] + candidates[0][2]))

    # Candidates that are nearly tied on the lowest pitch — there's no way to
    # distinguish them from low-E alone (Drop D vs Double Drop D vs DADGAD
    # vs Open D all have low D, for example). Use a chroma-based tiebreaker.
    tied = [c for c in candidates if abs(low_midi - (c[1][0] + c[2])) <= primary_low_delta + 0.3]
    if len(tied) > 1:
        best_label, best_midis, best_capo, disambig_note = _chroma_disambiguate(y, sr, tied)
    else:
        best_label, best_midis, best_capo = tied[0]
        disambig_note = ""
    effective_low = best_midis[0] + best_capo
    delta = low_midi - effective_low

    # Confidence based on how close the detected pitch is to the candidate.
    confidence = max(0.0, 1.0 - min(1.0, abs(delta) / 1.0))

    final_label = best_label if best_capo == 0 else f"{best_label}, capo {best_capo}"
    evidence = (
        f"lowest sustained pitch ≈ MIDI {low_midi:.2f} "
        f"({librosa.midi_to_note(low_midi, octave=True)}); "
        f"closest match: {final_label} "
        f"(effective low {effective_low}, offset {delta:+.2f} st)"
    )
    if disambig_note:
        evidence += f"; {disambig_note}"

    return Tuning(
        strings_midi=list(best_midis),
        capo=best_capo,
        label=final_label,
        confidence=round(confidence, 3),
        source="audio",
        evidence=evidence,
    )


def _chroma_disambiguate(
    y: np.ndarray, sr: int, tied_candidates: list[tuple[str, list[int], int]]
) -> tuple[str, list[int], int, str]:
    """Pick the best candidate among tunings that share an effective low pitch.

    For each candidate, computes how strongly its *open-string pitch classes*
    are represented in the audio's average chroma — the tuning whose open
    strings show up most in the song wins.

    Drop D vs Double Drop D for example: Drop D's open-string PCs include
    E (because high E open exists), Double Drop D's don't. If the song's
    average chroma has weak E, Double Drop D scores higher because removing
    that low-strength PC raises its averaged score.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        # Fall back to the first candidate if chroma fails.
        label, midis, capo = tied_candidates[0]
        return label, midis, capo, "chroma analysis failed; defaulted to first candidate"

    pc_strength = chroma.mean(axis=1)
    total = float(pc_strength.sum())
    if total <= 0:
        label, midis, capo = tied_candidates[0]
        return label, midis, capo, "chroma was silent; defaulted to first candidate"
    pc_strength = pc_strength / total  # normalize to fractions of total

    scored: list[tuple[float, tuple[str, list[int], int]]] = []
    for cand in tied_candidates:
        label, midis, capo = cand
        # Distinct pitch classes the open strings produce under this tuning.
        pcs = set((m + capo) % 12 for m in midis)
        # Sum of chroma strengths at those pitch classes — NOT normalized by
        # length. The previous avg-based scoring penalized tunings with more
        # open-string pitch classes (Drop D's 5 PCs vs DADGAD's 3); but a
        # tuning whose open strings cover more of the song's pitches is more
        # likely to be the right one. Using the sum rewards coverage.
        total = sum(float(pc_strength[pc]) for pc in pcs)
        scored.append((total, cand))

    scored.sort(key=lambda kv: kv[0], reverse=True)
    best_score, best_cand = scored[0]
    label, midis, capo = best_cand
    # Build the disambig note showing the runner-up for transparency.
    runner_up_score, runner_up_cand = scored[1] if len(scored) > 1 else (0.0, None)
    if runner_up_cand is not None:
        note = (
            f"chroma disambig picked {label} "
            f"(open-pcs total chroma {best_score:.3f}) over {runner_up_cand[0]} "
            f"({runner_up_score:.3f}) — {len(tied_candidates)} candidates tied on low pitch"
        )
    else:
        note = ""
    return label, midis, capo, note


# ---------------------------------------------------------------------------
# Transcription cross-check (capo/tuning contradiction veto)
# ---------------------------------------------------------------------------

# No tuning in the candidate library has an open string below MIDI 37
# ("Drop D, half-step down"). Transcribed pitches below this floor are model
# noise (cached MT3 output reaches down to MIDI 26) — never usable evidence.
_TRANSCRIPTION_JUNK_FLOOR_MIDI = 37

# Grace below the candidate's effective low pitch before a note counts as
# contradictory — absorbs single-semitone transcription wobble and bends.
_SUBFLOOR_GRACE_SEMITONES = 1

# A sub-floor note must sustain at least this long to count as evidence;
# anything shorter is an onset blip.
_SUBFLOOR_MIN_NOTE_S = 0.1

# Veto thresholds — ALL three must hold. Calibrated on cached videos
# (measured with the junk floor at 37):
#   9jswOBilMvA (wrong "Standard, capo 5"): 104 sustained sub-floor notes,
#     43.0 s total, 6.7 % of all notes            → must veto
#   wS_i91qxQYM (correct "Standard"): 4 sustained sub-floor notes (the MT3
#     octave-error Bass hallucinations at MIDI 35-36 are junk-filtered),
#     1.1 s, 0.06 % → must NOT veto
_SUBFLOOR_VETO_MIN_COUNT = 12
_SUBFLOOR_VETO_MIN_TOTAL_S = 8.0
_SUBFLOOR_VETO_MIN_FRACTION = 0.015

# A sub-floor pitch can only anchor the re-fit when it has enough sustained
# instances AND carries meaningful weight of the sub-floor evidence: it must
# be the modal sub-floor pitch, or hold at least the fraction below of the
# sustained sub-floor notes. A handful of MT3 octave errors must not drag
# the floor beneath the pitches the player actually sustained.
_FLOOR_SUPPORT_MIN_COUNT = 3
_FLOOR_SUPPORT_MIN_FRACTION = 0.15

# --- Sub-octave-phantom guard (symmetric with the contradiction veto) ------
# MT3/basic-pitch routinely emit a phantom note exactly an octave BELOW a
# real note (a sub-octave doubling). When a song has a real open D string
# (D3, MIDI 50 — present in BOTH standard and dropped tunings), the phantom
# D2 (MIDI 38) notes beneath those D3s make the audio detector conclude the
# lowest sustained pitch is D2 and pick a DROPPED tuning. This runs BEFORE
# the fret-stage overtone filter, so the floor needs its own protection here.
#
# A floor note is a phantom if it co-occurs (start within the window below)
# with a note exactly 12 semitones above it. The guard is INERT unless the
# chosen effective low string is BELOW standard low E (MIDI 40): a dropped
# floor is the only thing a phantom sub-octave can fabricate, and standard /
# capo floors (>= 40) must be left exactly as the contradiction veto leaves
# them (protects LBTD's E2 floor and Angie).
_DROPPED_FLOOR_CEILING_MIDI = 40  # guard only inspects floors strictly below this
_SUBOCTAVE_COOCCUR_WINDOW_S = 0.080  # +12 partner must start within ~80ms

# Need a real handful of floor notes before the doubled fraction is meaningful
# — a couple of doubled notes prove nothing about the tuning.
_MIN_FLOOR_NOTES = 8

# When re-fitting above a phantom floor, the upward scan skips pitches that are
# themselves sub-octave phantoms. A real low string heavily played inside
# octave / power chords (e.g. E2+E3) reads as >80% doubled too — without a
# bound, the scan would skip it and climb to an absurd floor (Standard, capo N).
# The library's dropped tunings all sit at D2=38, exactly TWO semitones below
# the standard low string E2=40, so the real floor (E2) is at phantom_floor+2.
# The skip is therefore allowed only STRICTLY below phantom_floor + this bound:
# pitches up to one semitone above the phantom can be artifacts, but a doubled
# pitch AT phantom_floor+2 (the standard low E) is a genuine octave-chorded
# string and must be accepted, not skipped.
_REFIT_MAX_SKIP_SEMITONES = 2

# Reject the dropped floor when this fraction of its notes are sub-octave
# doublings. Calibrated on cached videos (window 80ms):
#   S1GpQaD5yT0 (Crystal Ship, really Standard): D2 floor, 66 notes, 0.98
#     doubled by a simultaneous D3 → PHANTOM, must reject Drop D → Standard.
#   s_LAzeLbdLs (Song for George, really Double Drop D): D2 floor, 337 notes,
#     only 0.29 doubled → a REAL dropped low string, must NOT reject.
# 0.80 sits well clear of both ends (margin ~0.18 above George, ~0.18 below
# Crystal), so neither model noise on George nor a sparser-doubled phantom
# can land on the wrong side.
_SUBOCTAVE_PHANTOM_FRACTION = 0.80

# MT3's General-MIDI program for transcribed speech/singing — the only
# channel safe to drop wholesale (other non-guitar channels carry real
# guitar notes; see mt3.py).
_VOICE_PROGRAM = 65

# Confidence ceilings. A capo that no transcription has confirmed can never
# be 1.0 — reporting 1.0 from a single pyin statistic was the original bug.
# A vetoed-and-refit result is capped lower still: audio and transcription
# disagreed, so neither source is fully trustworthy.
_UNVERIFIED_CONFIDENCE_CAP = 0.85
_VETOED_CONFIDENCE_CAP = 0.75
# ...and floored: a vetoed re-fit is still presented as the answer, so it
# must never read as zero-confidence.
_VETOED_CONFIDENCE_FLOOR = 0.2


@dataclass
class _SubfloorEvidence:
    """Summary of transcribed notes below a candidate tuning's pitch floor."""

    all_count: int  # every note below floor-grace (incl. blips)
    sustained_count: int  # notes >= _SUBFLOOR_MIN_NOTE_S long
    total_s: float  # summed duration of the sustained notes
    fraction: float  # sustained_count / total notes examined
    supported_floor: int | None  # lowest sub-floor pitch with real support
    veto: bool


def _read_notes_file(path: Path) -> list[dict] | None:
    """Defensively parse a transcription JSON into [{pitch, start, end}, ...].

    The mt3 JSON schema is in flux (per-note channel tags / voice filtering
    are being added), so accept either ``{"notes": [...]}`` or a bare list,
    skip malformed records, and drop drum / singing-voice notes whenever the
    record carries enough metadata to identify them.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        # ValueError covers json.JSONDecodeError AND the UnicodeDecodeError
        # a truncated non-UTF-8 file (killed MT3 run) raises from read_text.
        return None
    raw = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return None
    notes: list[dict] = []
    for n in raw:
        if not isinstance(n, dict):
            continue
        try:
            pitch = int(n["pitch"])
            start = float(n["start"])
            end = float(n["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if n.get("is_drum"):
            continue
        program = n.get("program")
        if program is not None:
            try:
                if int(program) == _VOICE_PROGRAM:
                    continue
            except (TypeError, ValueError):
                pass
        name = str(n.get("instrument") or n.get("name") or "").lower()
        if "voice" in name or "vocal" in name:
            continue
        notes.append({"pitch": pitch, "start": start, "end": end})
    return notes or None


def load_transcribed_notes(paths: VideoPaths) -> tuple[list[dict], str] | None:
    """Best available transcription for cross-checking the tuning.

    Prefers ``notes.mt3.json`` (the default tab source), falling back to
    basic-pitch's ``notes.json``. Returns ``(notes, source_name)`` or None
    when neither file yields usable notes.
    """
    for path, source in ((paths.notes_mt3_json, "mt3"), (paths.notes_json, "basic_pitch")):
        notes = _read_notes_file(path)
        if notes:
            return notes, source
    return None


def _subfloor_evidence(notes: list[dict], effective_low: int) -> _SubfloorEvidence:
    """Collect the notes that contradict an effective lowest sounding pitch."""
    threshold = effective_low - _SUBFLOOR_GRACE_SEMITONES
    sub = [n for n in notes if _TRANSCRIPTION_JUNK_FLOOR_MIDI <= n["pitch"] < threshold]
    sustained = [n for n in sub if (n["end"] - n["start"]) >= _SUBFLOOR_MIN_NOTE_S]
    total_s = float(sum(n["end"] - n["start"] for n in sustained))
    fraction = len(sustained) / len(notes) if notes else 0.0

    # The lowest sub-floor pitch with enough sustained instances — and a
    # meaningful share of the sub-floor evidence — to anchor a re-fit. None
    # when the sub-floor notes are scattered one-offs.
    supported_floor: int | None = None
    by_pitch = Counter(n["pitch"] for n in sustained)
    if by_pitch:
        modal_count = max(by_pitch.values())
        min_share = _FLOOR_SUPPORT_MIN_FRACTION * len(sustained)
        for pitch in sorted(by_pitch):
            count = by_pitch[pitch]
            if count < _FLOOR_SUPPORT_MIN_COUNT:
                continue
            if count == modal_count or count >= min_share:
                supported_floor = pitch
                break

    veto = (
        len(sustained) >= _SUBFLOOR_VETO_MIN_COUNT
        and total_s >= _SUBFLOOR_VETO_MIN_TOTAL_S
        and fraction >= _SUBFLOOR_VETO_MIN_FRACTION
        and supported_floor is not None
    )
    return _SubfloorEvidence(
        all_count=len(sub),
        sustained_count=len(sustained),
        total_s=total_s,
        fraction=fraction,
        supported_floor=supported_floor,
        veto=veto,
    )


@dataclass
class _PhantomFloorEvidence:
    """Summary of whether a dropped tuning's floor pitch is sub-octave noise."""

    floor_midi: int  # the candidate's effective low pitch examined
    floor_count: int  # notes sitting at exactly that pitch
    doubled_count: int  # of those, ones with a +12 partner within the window
    doubled_fraction: float  # doubled_count / floor_count
    phantom: bool  # floor is a sub-octave artifact → reject the dropped tuning


def _phantom_floor_evidence(notes: list[dict], floor_midi: int) -> _PhantomFloorEvidence:
    """Decide whether the notes at ``floor_midi`` are sub-octave doublings.

    A floor note is treated as phantom when another note exactly 12 semitones
    above it starts within ``_SUBOCTAVE_COOCCUR_WINDOW_S`` — the model's
    octave-below doubling of a real note. When (almost) every floor note is
    such a doubling there is no genuine string at that pitch.
    """
    at_floor = [n for n in notes if n["pitch"] == floor_midi]
    partner_starts = sorted(n["start"] for n in notes if n["pitch"] == floor_midi + 12)

    doubled = 0
    for n in at_floor:
        # A +12 partner whose onset is within the window of this note's onset.
        lo = n["start"] - _SUBOCTAVE_COOCCUR_WINDOW_S
        hi = n["start"] + _SUBOCTAVE_COOCCUR_WINDOW_S
        idx = bisect_left(partner_starts, lo)
        if idx < len(partner_starts) and partner_starts[idx] <= hi:
            doubled += 1

    floor_count = len(at_floor)
    fraction = doubled / floor_count if floor_count else 0.0
    phantom = floor_count >= _MIN_FLOOR_NOTES and fraction > _SUBOCTAVE_PHANTOM_FRACTION
    return _PhantomFloorEvidence(
        floor_midi=floor_midi,
        floor_count=floor_count,
        doubled_count=doubled,
        doubled_fraction=fraction,
        phantom=phantom,
    )


def _refit_above_phantom_floor(
    phantom_floor: int, notes: list[dict], original: Tuning
) -> tuple[str, list[int], int]:
    """Re-fit a dropped tuning whose floor was sub-octave noise.

    The real floor is the lowest pitch the player actually sustains once the
    phantom sub-octave is removed: scan upward from one semitone above the
    phantom floor for the first non-junk pitch that has real support
    (>= _FLOOR_SUPPORT_MIN_COUNT sustained instances) AND is not itself a
    sub-octave phantom of a note 12 st higher. For Crystal Ship that skips the
    14 residual phantom D2(38)s and lands on Standard's E2(40, 52 instances).

    The phantom-skip is BOUNDED to _REFIT_MAX_SKIP_SEMITONES above the phantom
    floor: a real low string that is itself heavily octave-doubled (E2 struck
    inside E2+E3 power chords reads >80% doubled too) must not be skipped, or
    the scan would climb past the true floor to an absurd capo'd standard. The
    fabricating phantom is always one semitone below the standard low string in
    the library's dropped tunings, so the real floor is within the bound.
    """
    sustained = [n for n in notes if (n["end"] - n["start"]) >= _SUBFLOOR_MIN_NOTE_S]
    by_pitch = Counter(
        n["pitch"]
        for n in sustained
        if _TRANSCRIPTION_JUNK_FLOOR_MIDI <= n["pitch"] and n["pitch"] > phantom_floor
    )
    real_floor: int | None = None
    skip_ceiling = phantom_floor + _REFIT_MAX_SKIP_SEMITONES
    for pitch in sorted(by_pitch):
        if by_pitch[pitch] < _FLOOR_SUPPORT_MIN_COUNT:
            continue
        # Skip a pitch that is itself a sub-octave phantom of a +12 partner —
        # but only strictly within the bound above the phantom floor. AT or past
        # the bound (the standard low string E2=40 sits exactly there for a D2=38
        # phantom) a heavily-doubled pitch is a genuine (octave-chorded) low
        # string, so accept it as the real floor instead of climbing past it.
        if pitch < skip_ceiling and _phantom_floor_evidence(sustained, pitch).phantom:
            continue
        real_floor = pitch
        break
    # Fall back to standard low E when nothing else carries support: a phantom
    # dropped floor with no real low string below the standard one IS standard.
    if real_floor is None:
        real_floor = STANDARD_TUNING[0]
    return _refit_to_floor(real_floor, original)


def _refit_to_floor(floor_midi: int, original: Tuning) -> tuple[str, list[int], int]:
    """Pick the (tuning, capo) candidate whose effective low pitch best
    explains the transcription's supported floor. Tie-breaks: smaller capo,
    then keeping the originally-detected string set."""

    def score(cand: tuple[str, list[int], int]) -> tuple[float, int, int]:
        _, midis, capo = cand
        delta = abs(floor_midi - (min(midis) + capo))
        same_strings = 0 if list(midis) == list(original.strings_midi) else 1
        return (delta, capo, same_strings)

    return min(_AUDIO_CANDIDATES, key=score)


def _reject_phantom_floor(
    tuning: Tuning,
    notes: list[dict],
    notes_source: str,
    phantom: _PhantomFloorEvidence,
    verification: dict,
) -> Tuning:
    """Re-fit a dropped tuning whose low floor was a sub-octave phantom.

    Mirrors the contradiction-veto re-fit, but in the OPPOSITE direction:
    instead of lowering the floor for impossible sub-floor notes, it RAISES
    the floor off a phantom that fabricated a dropped string.
    """
    best_label, best_midis, best_capo = _refit_above_phantom_floor(
        phantom.floor_midi, notes, tuning
    )
    new_low = min(best_midis) + best_capo
    # The guard only fires when doubled_fraction > _SUBOCTAVE_PHANTOM_FRACTION
    # (0.80), so the max(_VETOED_CONFIDENCE_FLOOR, ...) of the contradiction
    # path can never bind here; a phantom re-fit is always capped to exactly
    # _VETOED_CONFIDENCE_CAP (0.75). Stated as a literal for clarity.
    confidence = _VETOED_CONFIDENCE_CAP
    final_label = best_label if best_capo == 0 else f"{best_label}, capo {best_capo}"
    verification.update(
        {
            "veto": True,
            "vetoed_label": tuning.label,
            "suboctave_phantom_floor_midi": phantom.floor_midi,
            "refit_floor_midi": new_low,
        }
    )
    return Tuning(
        strings_midi=list(best_midis),
        capo=best_capo,
        label=final_label,
        confidence=round(confidence, 3),
        source=f"{tuning.source}+suboctave-phantom-veto",
        evidence=(
            f"{tuning.evidence}; VETOED '{tuning.label}': its low floor MIDI "
            f"{phantom.floor_midi} is a sub-octave phantom — "
            f"{phantom.doubled_count}/{phantom.floor_count} "
            f"({phantom.doubled_fraction:.1%}) of those {notes_source} notes are "
            f"doubled by a simultaneous note 12 st above; re-fit to real floor "
            f"MIDI {new_low} → {final_label}"
        ),
        verification=verification,
    )


def verify_against_transcription(tuning: Tuning, paths: VideoPaths) -> Tuning:
    """Cross-check a chosen (tuning, capo) against the transcribed notes.

    The audio heuristic can pick a capo the transcription flatly contradicts:
    on 9jswOBilMvA pyin missed the low E2s and produced "Standard, capo 5" at
    confidence 1.0 while notes.mt3.json held 96+ notes below the capo-5 floor
    — every one of which was then silently deleted downstream (470 notes,
    all the E and F#m chords). Sustained, repeated notes below the
    candidate's effective lowest pitch are physically impossible, so they
    VETO the candidate: we re-fit to the transcription's own supported floor
    and reduce confidence. When no transcription exists yet, the result is
    kept but marked unverified with confidence capped below 1.0.

    Always records a ``verification`` dict on the returned Tuning so
    tuning.json says WHY the result stands.
    """
    loaded = load_transcribed_notes(paths)
    if loaded is None:
        # Return a copy and guard the evidence suffix — verifying the same
        # Tuning twice must not mutate the input or stack the suffix.
        unverified_note = "; capo unverified (no transcription available at detection time)"
        new_evidence = tuning.evidence
        if unverified_note not in new_evidence:
            new_evidence += unverified_note
        return replace(
            tuning,
            confidence=min(tuning.confidence, _UNVERIFIED_CONFIDENCE_CAP),
            evidence=new_evidence,
            verification={
                "checked": False,
                "reason": "no transcription available at detection time",
            },
        )

    notes, notes_source = loaded
    effective_low = min(tuning.effective_open_pitches())
    evidence = _subfloor_evidence(notes, effective_low)
    verification = {
        "checked": True,
        "notes_source": notes_source,
        "notes_examined": len(notes),
        "effective_low_midi": effective_low,
        "subfloor_count": evidence.all_count,
        "subfloor_sustained_count": evidence.sustained_count,
        "subfloor_total_s": round(evidence.total_s, 2),
        "subfloor_fraction": round(evidence.fraction, 4),
        "veto": False,
    }

    if not evidence.veto:
        # Sub-octave-phantom guard (symmetric with the contradiction veto):
        # the audio detector can pick a DROPPED tuning whose only evidence for
        # the low floor is phantom sub-octave doublings of a real note an
        # octave up. Inert unless the chosen floor is below standard low E.
        if effective_low < _DROPPED_FLOOR_CEILING_MIDI:
            phantom = _phantom_floor_evidence(notes, effective_low)
            verification.update(
                {
                    "suboctave_floor_count": phantom.floor_count,
                    "suboctave_doubled_fraction": round(phantom.doubled_fraction, 4),
                    "suboctave_phantom_rejected": phantom.phantom,
                }
            )
            if phantom.phantom:
                return _reject_phantom_floor(tuning, notes, notes_source, phantom, verification)
        return replace(tuning, verification=verification)

    # Contradiction: re-fit among candidates that can actually produce the
    # transcription's supported floor pitch.
    assert evidence.supported_floor is not None  # guaranteed when veto is True
    best_label, best_midis, best_capo = _refit_to_floor(evidence.supported_floor, tuning)
    new_low = min(best_midis) + best_capo
    delta = evidence.supported_floor - new_low
    confidence = min(
        _VETOED_CONFIDENCE_CAP, max(_VETOED_CONFIDENCE_FLOOR, 1.0 - min(1.0, abs(delta)))
    )
    final_label = best_label if best_capo == 0 else f"{best_label}, capo {best_capo}"

    residual = _subfloor_evidence(notes, new_low)
    verification.update(
        {
            "veto": True,
            "vetoed_label": tuning.label,
            "supported_floor_midi": evidence.supported_floor,
            "residual_subfloor_sustained_count": residual.sustained_count,
        }
    )
    return Tuning(
        strings_midi=list(best_midis),
        capo=best_capo,
        label=final_label,
        confidence=round(confidence, 3),
        source=f"{tuning.source}+contradiction-veto",
        evidence=(
            f"{tuning.evidence}; VETOED '{tuning.label}': {evidence.sustained_count} sustained "
            f"transcribed notes ({evidence.total_s:.1f}s, {evidence.fraction:.1%} of "
            f"{len(notes)} {notes_source} notes) sit below its effective low "
            f"{effective_low}; re-fit to transcribed floor MIDI "
            f"{evidence.supported_floor} → {final_label} (offset {delta:+d} st)"
        ),
        verification=verification,
    )
