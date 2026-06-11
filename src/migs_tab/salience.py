"""Audio-evidence (salience) scoring: does the audio actually support a note?

YourMT3 emits a constant velocity of 100 for every note, so none of the
velocity-gated artifact filters in fret.py can work on MT3 output, and
nothing downstream can tell a confidently-played chord tone from a phantom
transcription artifact. This module derives that evidence directly from the
separated guitar stem:

- ``note_salience`` — per-note percentile rank of the claimed pitch's CQT bin
  in the 30-200 ms post-onset window. A feasibility spike on the Angie cache
  (wS_i91qxQYM) showed this beats ±1-2-semitone shifts of the same note 93.6%
  of the time when used *comparatively* (see ``compare_pitches_at_onset``).
- ``octave_artifact_flags`` — flags probable phantom overtones (+12 of a real
  note in the same onset cluster) via the energy ratio of the two bins.
- ``pseudo_velocity`` — maps post-onset CQT magnitude to a 0-127 velocity
  proxy so velocity-gated filters can be revived for MT3 notes.
- ``section_score`` — section-level verification: Karplus-Strong-synthesize
  the claimed notes and compare frame-wise CQT-chroma cosine against the real
  stem at absolute timestamps. The spike measured 0.893 for the verified-good
  tab vs 0.586 for random pitches, degrading monotonically with corruption.
  Free-alignment DTW was tested and REJECTED: it ranked unrelated audio above
  a 15%-corrupted tab. Do not add DTW here.

Downstream consumers: fret.py calls ``octave_artifact_flags`` and
``pseudo_velocity`` pre-Viterbi; verify.py — planned (wave 2) — will call
``note_salience``, ``compare_pitches_at_onset`` and ``section_score`` per
section.

All timestamps in this module are ABSOLUTE seconds in the source video.
``load_stem_window`` + a ``window_start`` argument keep the audio I/O bounded
without making callers juggle relative offsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Analysis sample rate. The stems are 44.1 kHz stereo; 22.05 kHz mono is
# plenty for pitch bins up to A6 and halves every CQT below.
DEFAULT_SR = 22050
_HOP_LENGTH = 512

# CQT pitch coverage: MIDI 36 (C2) .. 93 (A6), one bin per semitone. A guitar
# in any tuning this project handles spans ~36-88, and cached MT3 output
# claims pitches 26-100 — covering a bit beyond the playable range keeps
# octave errors visible as *real bins with low energy* instead of silently
# falling off the edge of the transform.
PITCH_MIN_MIDI = 36
PITCH_MAX_MIDI = 93
_N_BINS = PITCH_MAX_MIDI - PITCH_MIN_MIDI + 1
_BINS_PER_OCTAVE = 12

# Post-onset analysis window: skip the broadband attack transient (first
# 30 ms), average over the early decay where the struck pitch dominates.
# Same 30-200 ms window the spike validated.
POST_ONSET_START = 0.030
POST_ONSET_END = 0.200

# librosa.cqt needs ~8192 samples at this sr/fmin (measured: 6144 fails,
# 8192 works). Pad anything shorter with trailing zeros.
_MIN_CQT_SAMPLES = 8192

# Octave-artifact energy-ratio threshold, CALIBRATED on cache/wS_i91qxQYM
# (Angie; chroma-verified windows 1.0-76.8 s and 1107-1125 s):
#
#   Proven phantoms, mean post-onset E(p)/E(p-12) per strum:
#     B4=71 over real B3 in the open-E7 cluster 903.2-904.8 s:
#         0.016-0.130 (6 strums)
#     G#4=68 over real G#3, same cluster: 0.132-0.317 (6 strums)
#     A5=81 over real A4 at 828.42-829.09 s: 0.370-0.531 (3 onsets)
#   Genuine octave pairs (28 pairs from clusters matching FULL known open
#   voicings — Am 45/57 + 52/64, C 48/60 + 52/64, F 53/65 + 48/60, Dm 50/62 —
#   measured at the upper note's own onset):
#         min 0.154, p5 0.242, p25 0.481, median 0.637, max 8.2
#
#   At 0.20 the flag catches 9/15 phantom measurements (all six B4 strums,
#   three of six G#4 strums) while false-flagging 1/28 genuine pairs (3.6%).
#   The distributions OVERLAP above ~0.25: A5-style louder overtones
#   (0.37-0.53) cannot be separated from genuine octaves by this ratio alone
#   (genuine p25 is 0.48), so the threshold is deliberately precision-first —
#   downstream deletes flagged notes, and over-deletion was the original sin
#   this project is recovering from. Use compare_pitches_at_onset /
#   note_salience as the second opinion for the louder cases.
OCTAVE_ARTIFACT_RATIO = 0.20

# Pseudo-velocity mapping: post-onset mean CQT magnitude at the claimed bin,
# expressed in dB below the loudest bin of the analysis window, mapped
# linearly onto 0-127 over a 60 dB range:
#     velocity = round(127 * clip(1 + dB/60, 0, 1))
# So the window's loudest moment ≈ 127, anything ≥60 dB down (or silent) = 0.
# Per-window normalization means values are comparable WITHIN a window
# (which is what the within-cluster ratio filters in fret.py need), not
# across windows.
_PSEUDO_VELOCITY_DB_RANGE = 60.0

# Velocity assigned to events that cannot be measured (outside the loaded
# window, or pitch outside the CQT range). Neutral: above the chord-context
# filter's floor of 55, below any accent — never auto-dropped on velocity
# grounds alone.
PSEUDO_VELOCITY_UNSCORED = 64

# Karplus-Strong synthesis (lifted from the validated spike): feedback gain,
# extra ring-out beyond the notated duration, duration clamp, default seed
# for the excitation noise.
_KS_GAIN = 0.995
_KS_RING_EXTRA = 0.25
_KS_MIN_DUR = 0.4
_KS_MAX_DUR = 2.5
_KS_SEED = 7


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CQTContext:
    """One window's CQT magnitudes plus the bookkeeping to query it by
    absolute timestamp and MIDI pitch.

    Build once per analysis window via :func:`compute_cqt_context` and pass
    it to note_salience / compare_pitches_at_onset / octave_artifact_flags /
    pseudo_velocity so the (expensive) CQT runs exactly once. note_salience
    and pseudo_velocity build their own context when none is passed.
    """

    magnitudes: np.ndarray  # (n_bins, n_frames) CQT magnitude
    frame_times: np.ndarray  # ABSOLUTE time (s) of each frame
    window_start: float  # absolute start of the analysed audio
    pitch_min: int = PITCH_MIN_MIDI

    def bin_for_pitch(self, midi_pitch: int) -> int | None:
        """Row index for a MIDI pitch, or None if outside the CQT range."""
        b = int(midi_pitch) - self.pitch_min
        if 0 <= b < self.magnitudes.shape[0]:
            return b
        return None

    def post_onset_column(self, onset: float) -> np.ndarray | None:
        """Mean magnitude per bin over [onset+30ms, onset+200ms] (absolute
        seconds). Frames falling outside the window are simply absent —
        an onset near the window's edge is scored from whatever frames
        exist. Returns None when no frames are available at all."""
        mask = (self.frame_times >= onset + POST_ONSET_START) & (
            self.frame_times <= onset + POST_ONSET_END
        )
        if not mask.any():
            return None
        return self.magnitudes[:, mask].mean(axis=1)


# ---------------------------------------------------------------------------
# Audio loading + CQT
# ---------------------------------------------------------------------------


def load_stem_window(
    stem_path: str | Path, start: float, end: float, sr: int = DEFAULT_SR
) -> tuple[np.ndarray, int]:
    """Load [start, end) seconds of a stem as mono float at ``sr``.

    Uses librosa's offset/duration so only the requested slice is decoded —
    stems run 30+ minutes and must never be loaded whole.

    Raises ValueError if ``start`` is negative (negative offsets silently
    break soundfile's fast path; fail loudly at the API boundary instead).
    """
    start = float(start)
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    duration = max(0.0, float(end) - start)
    y, sr_out = librosa.load(str(stem_path), sr=sr, mono=True, offset=start, duration=duration)
    return y, int(sr_out)


def compute_cqt_context(y: np.ndarray, sr: int, window_start: float) -> CQTContext:
    """Compute the semitone CQT (MIDI 36-93) for one audio window.

    ``window_start`` is the absolute time of ``y[0]``; the returned context
    answers queries in absolute seconds. Audio shorter than the lowest CQT
    filter is zero-padded.
    """
    if len(y) < _MIN_CQT_SAMPLES:
        y = np.pad(y, (0, _MIN_CQT_SAMPLES - len(y)))
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=_HOP_LENGTH,
            fmin=librosa.midi_to_hz(PITCH_MIN_MIDI),
            n_bins=_N_BINS,
            bins_per_octave=_BINS_PER_OCTAVE,
        )
    )
    frame_times = (
        librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=_HOP_LENGTH) + window_start
    )
    return CQTContext(magnitudes=C, frame_times=frame_times, window_start=window_start)


# ---------------------------------------------------------------------------
# Per-note salience
# ---------------------------------------------------------------------------


def salience_at(ctx: CQTContext, onset: float, midi_pitch: int) -> float | None:
    """Percentile rank of ``midi_pitch``'s bin in the mean post-onset CQT
    column: 1.0 = the claimed pitch is the strongest thing ringing, ~0.5 = no
    better than chance. None when the event can't be scored (onset outside
    the context window, pitch outside the CQT range, or zero energy at the
    claimed bin — digital silence would otherwise rank every pitch a perfect
    1.0 and falsely "verify" a note claimed over nothing)."""
    b = ctx.bin_for_pitch(midi_pitch)
    if b is None:
        return None
    col = ctx.post_onset_column(onset)
    if col is None or col[b] <= 0.0:
        return None
    return float((col <= col[b]).mean())


def note_salience(
    events: Sequence[tuple[float, int]],
    y: np.ndarray,
    sr: int,
    window_start: float,
    ctx: CQTContext | None = None,
) -> list[float | None]:
    """Salience score per event over one shared CQT.

    ``events`` are (onset_seconds_absolute, midi_pitch) pairs; ``y`` is the
    audio starting at ``window_start`` (see :func:`load_stem_window`). The
    result is aligned with ``events``; unscorable events get None. Pass a
    prebuilt ``ctx`` (for the same ``y``/``sr``/``window_start``) to reuse
    one CQT across scoring functions; when None it is computed here.
    """
    if ctx is None:
        ctx = compute_cqt_context(y, sr, window_start)
    return [salience_at(ctx, onset, pitch) for onset, pitch in events]


def compare_pitches_at_onset(
    onset: float,
    candidate_pitches: Sequence[int],
    cqt_context: CQTContext,
) -> list[tuple[int, float]]:
    """Rank candidate pitches for one onset by salience, best first.

    This is the comparative form the spike validated (true pitch outranks
    its own ±1-2-semitone shifts 93.6% of the time) — use it to adjudicate
    between a chosen pitch and its alternatives rather than thresholding
    absolute salience. Unscorable candidates — out of range, out of window,
    or claimed over a silent bin (salience_at returns None) — rank last with
    score 0.0 (real saliences are always > 0).
    """
    scored = []
    for pitch in candidate_pitches:
        s = salience_at(cqt_context, onset, int(pitch))
        scored.append((int(pitch), 0.0 if s is None else s))
    scored.sort(key=lambda ps: ps[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Octave-artifact flagging
# ---------------------------------------------------------------------------


def octave_artifact_flags(
    cluster_events: Sequence[tuple[float, int]],
    cqt_context: CQTContext,
    ratio_threshold: float = OCTAVE_ARTIFACT_RATIO,
) -> list[bool]:
    """Flag probable phantom overtones within one onset cluster.

    For each event with pitch ``p`` whose cluster also claims ``p - 12``
    (the 2nd harmonic's fundamental), compare the mean post-onset CQT energy
    at the two bins — both measured at the candidate's own onset. When
    ``E(p) / E(p-12) < ratio_threshold`` the upper note is energy-consistent
    with being nothing but the lower note's overtone, and gets flagged.

    Events without a sounding p-12 partner, with pitches outside the CQT
    range, or outside the context window are never flagged. The returned
    list is aligned with ``cluster_events``.
    """
    pitch_set = {int(p) for _, p in cluster_events}
    flags: list[bool] = []
    for onset, pitch in cluster_events:
        pitch = int(pitch)
        if pitch - 12 not in pitch_set:
            flags.append(False)
            continue
        b_hi = cqt_context.bin_for_pitch(pitch)
        b_lo = cqt_context.bin_for_pitch(pitch - 12)
        if b_hi is None or b_lo is None:
            flags.append(False)
            continue
        col = cqt_context.post_onset_column(onset)
        if col is None or col[b_lo] <= 0.0:
            flags.append(False)
            continue
        flags.append(float(col[b_hi]) / float(col[b_lo]) < ratio_threshold)
    return flags


# ---------------------------------------------------------------------------
# Pseudo-velocity
# ---------------------------------------------------------------------------


def pseudo_velocity(
    events: Sequence[tuple[float, int]],
    y: np.ndarray,
    sr: int,
    window_start: float,
    ctx: CQTContext | None = None,
) -> list[int]:
    """Audio-derived 0-127 velocity proxy per event.

    Post-onset mean CQT magnitude at the claimed bin, in dB below the
    window's loudest bin, mapped linearly over ``_PSEUDO_VELOCITY_DB_RANGE``
    (see the constant's comment for the exact formula). Monotone in source
    amplitude. Unscorable events get ``PSEUDO_VELOCITY_UNSCORED``. Pass a
    prebuilt ``ctx`` (for the same ``y``/``sr``/``window_start``) to reuse
    one CQT across scoring functions; when None it is computed here.
    """
    if ctx is None:
        ctx = compute_cqt_context(y, sr, window_start)
    ref = float(ctx.magnitudes.max())
    out: list[int] = []
    for onset, pitch in events:
        b = ctx.bin_for_pitch(int(pitch))
        col = ctx.post_onset_column(onset) if b is not None else None
        if b is None or col is None or ref <= 0.0:
            out.append(PSEUDO_VELOCITY_UNSCORED)
            continue
        e = float(col[b])
        if e <= 0.0:
            out.append(0)
            continue
        db = 20.0 * np.log10(e / ref)
        vel = int(round(127.0 * float(np.clip(1.0 + db / _PSEUDO_VELOCITY_DB_RANGE, 0.0, 1.0))))
        out.append(vel)
    return out


# ---------------------------------------------------------------------------
# Section-level synthesize-and-compare
# ---------------------------------------------------------------------------


def _karplus_strong_pluck(f0: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Block-vectorized Karplus-Strong pluck (validated in the spike)."""
    n_total = max(1, int(dur * sr))
    period = max(2, int(round(sr / f0)))
    buf = rng.uniform(-1.0, 1.0, period)
    blocks = []
    made = 0
    while made < n_total:
        blocks.append(buf.copy())
        buf = _KS_GAIN * 0.5 * (buf + np.roll(buf, -1))
        made += period
    y = np.concatenate(blocks)[:n_total]
    # Short attack/release ramps to avoid clicks.
    a = min(64, n_total)
    y[:a] *= np.linspace(0, 1, a)
    y[-a:] *= np.linspace(1, 0, a)
    return y


def karplus_strong_render(
    events_with_duration: Sequence[tuple[float, float, int]],
    sr: int = DEFAULT_SR,
    seed: int = _KS_SEED,
) -> np.ndarray:
    """Render (onset_seconds, duration_seconds, midi_pitch) events to audio.

    Onsets are relative to the returned buffer's start. Each note rings
    ``_KS_RING_EXTRA`` beyond its notated duration (clamped to
    [_KS_MIN_DUR, _KS_MAX_DUR]) like a real undamped string. Deterministic
    for a given seed; peak-normalized to 0.9.
    """
    if not events_with_duration:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    total_dur = max(
        t + float(np.clip(d + _KS_RING_EXTRA, _KS_MIN_DUR, _KS_MAX_DUR))
        for t, d, _ in events_with_duration
    )
    mix = np.zeros(int(total_dur * sr) + sr, dtype=np.float64)
    for t, d, pitch in events_with_duration:
        dur = float(np.clip(d + _KS_RING_EXTRA, _KS_MIN_DUR, _KS_MAX_DUR))
        y = _karplus_strong_pluck(librosa.midi_to_hz(pitch), dur, sr, rng)
        i0 = max(0, int(t * sr))
        mix[i0 : i0 + len(y)] += y
    mix = mix[: int(total_dur * sr)]
    peak = np.max(np.abs(mix))
    return (mix / peak * 0.9) if peak > 0 else mix


def _framewise_chroma_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-frame cosine similarity of two chromagrams at ABSOLUTE frame
    positions (no alignment — free-alignment DTW was tested and rejected)."""
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    ok = den > 1e-9
    if not ok.any():
        return 0.0
    return float((num[ok] / den[ok]).mean())


def section_score(
    events: Sequence[tuple[float, float, int]],
    stem_path: str | Path,
    start: float,
    end: float,
    sr: int = DEFAULT_SR,
) -> float:
    """Section-level tab-vs-audio agreement in [0, 1]-ish (chroma cosine).

    ``events`` are (onset_seconds_absolute, duration_seconds, midi_pitch);
    only those with onsets inside [start, end) are used. The claimed notes
    are Karplus-Strong-synthesized and compared to the real stem frame-by-
    frame at absolute timestamps. Spike reference points on Angie's verified
    window: real tab 0.893, 15% corrupted 0.843, 40% corrupted 0.743,
    random pitches 0.586, unrelated window 0.605 — treat ~0.85+ as verified,
    monotone decay below.
    """
    y_real, sr = load_stem_window(stem_path, start, end, sr=sr)
    if len(y_real) == 0:
        return 0.0
    local = [(t - start, d, int(p)) for t, d, p in events if start <= t < end]
    if not local:
        return 0.0
    y_syn = karplus_strong_render(local, sr=sr)
    # Match lengths so the chromagrams cover the same absolute span.
    n = len(y_real)
    if len(y_syn) < n:
        y_syn = np.pad(y_syn, (0, n - len(y_syn)))
    else:
        y_syn = y_syn[:n]
    chroma_real = librosa.feature.chroma_cqt(y=y_real, sr=sr, hop_length=_HOP_LENGTH)
    chroma_syn = librosa.feature.chroma_cqt(y=y_syn, sr=sr, hop_length=_HOP_LENGTH)
    return _framewise_chroma_cosine(chroma_syn, chroma_real)
