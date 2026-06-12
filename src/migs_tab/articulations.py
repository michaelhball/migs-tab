"""Audio-gated articulation detection: hammer-ons, pull-offs, slides, bends,
natural harmonics.

Every detector here is gated on measured stem audio, not on interval/timing
heuristics alone — a tab sprayed with bogus h/p marks is worse than one with
none, so precision beats recall throughout. Detection runs on frets.json
note records (post-Viterbi), referencing notes by their global ``note_index``.

Three independent detectors:

1. **Hammer/pull/slide** — candidate = consecutive same-string notes with a
   small inter-onset gap. The audio gate measures the second note's *attack
   transient*: a picked note carries a broadband click that survives mostly
   ABOVE 2 kHz, which the semitone CQT (top bin A6 ≈ 1.76 kHz) cannot see —
   raw CQT flux failed to separate the calibration pairs (hammer ratios
   0.30-1.90 vs picked 0.33-1.43, fully overlapped). What worked: HPSS on a
   2048-bin STFT, then peak percussive energy >= 2 kHz around the onset,
   normalized by mean post-onset harmonic energy (``_AttackMeasurer.attack``).
   Absolute values do not transfer across playing styles (LBTD pick attacks
   measure 0.9-2.7 where fingerstyle Angie picks measure 0.2-0.6), so the
   gate combines the absolute value with a within-window percentile and the
   ratio to the first note's attack (see _ATTACK_* constants).

2. **Bend** — candidate = same-string pair 1-2 semitones apart whose second
   onset falls inside (or just past) the first note's sustain. Evidence is a
   high-resolution (3 bins/semitone) CQT pitch track over the pair's span:
   the struck pitch must sit at the source bin early and migrate to a
   sustained tail >= 0.75 semitones up. Calibrated on Angie's 1451.0 s
   "slinky" Bb bend: struck centroid 58.02, tail 59.04 sustained 0.45 s
   (the mid-flight B3 transcription artifact becomes a hidden member note).

3. **Natural harmonic** — deliberately conservative: only 12th-fret node
   candidates (pitch = open string + 12), long (>= 0.8 s), isolated
   (singleton cluster), and spectrally consistent with a touched harmonic.
   On the labeled Angie t=1.0 s harmonic vs the 86 other long singleton
   candidates in the same video, the gate below passes exactly the labeled
   note (purity 0.307 vs next-best 0.185; see _HARMONIC_* constants).
   Attack is NOT a harmonic discriminator (the labeled harmonic measures
   attack 2.23 — it is plucked sharply); the sustain spectrum is.

The output contract (consumed by render.py) — entries reference the global
``note_index`` of frets.json note records:

- {"type": "hammer"|"pull"|"slide", "from_note_index", "note_index",
   "string", "from_fret", "to_fret", "evidence": {...}}
- {"type": "bend", "note_index", "string", "fret", "target_semitones",
   "member_note_indices", "evidence": {...}}  # members must be HIDDEN
- {"type": "harmonic", "note_index", "open_string", "node_fret",
   "evidence": {...}}  # re-string to the open string's 12th-fret node
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np

from .salience import CQTContext, compute_cqt_context, load_stem_window

# ---------------------------------------------------------------------------
# Shared analysis constants
# ---------------------------------------------------------------------------

# Attack analysis windows mirror fret.py's evidence pass: 80 s windows with a
# 1 s tail pad so onsets near a boundary keep their full post-onset frames.
DEFAULT_WINDOW_SECONDS = 80.0
_WINDOW_PAD = 1.0

_SR = 22050
_N_FFT = 2048
_HOP = 512

# Percussive attack band floor. The pick click is broadband but the part the
# guitar's own harmonics can't imitate lives above ~2 kHz; restricting the
# percussive (HPSS) energy to >= 2 kHz gave the cleanest hammer-vs-picked
# separation on the LBTD calibration riff.
_ATTACK_BAND_HZ = 2000.0

# Attack sampling window around an onset (peak percussive) and the post-onset
# span whose mean harmonic energy normalizes it. MT3 onsets are good to
# ~±20 ms; 70 ms of lookahead catches late-blooming attacks without leaking
# into the next 16th note at practical tempos.
_ATTACK_PRE = 0.02
_ATTACK_POST = 0.07
_ATTACK_HARM_START = 0.03
_ATTACK_HARM_END = 0.20

# ---------------------------------------------------------------------------
# Hammer / pull / slide gates
# ---------------------------------------------------------------------------

# Max inter-onset gap for a legato pair. The labeled LBTD slow-demo hammers
# (A|0h3 at 61.62→61.99, D|1h2 at 62.78→63.19) sit at 0.37-0.41 s — slower
# than the design sketch's 0.30 because teaching demos are slow — so the
# timing gate is loose and the audio gate does the real filtering.
_MAX_LEGATO_GAP = 0.45

# Fret-delta ranges. 1-2 fretted = hammer/pull, 3-7 fretted-to-fretted =
# slide. Pairs involving an OPEN string cannot be slides (there is no finger
# to slide), so open-source ascents and open-target descents up to 4 frets
# are typed hammer/pull instead — the community-tab figures A|0h3 and e|3p0
# are exactly such pairs — and open-endpoint moves wider than that are no
# articulation at all (the first run emitted "slides" like 0->7, which are
# just soft position changes).
_HP_MAX_DELTA = 2
_OPEN_LEGATO_MAX_DELTA = 4
_SLIDE_MIN_DELTA = 3
_SLIDE_MAX_DELTA = 7

# A legato target must be an (almost) lone event: a hammer/pull/slide is a
# one-finger action, not a strum. The first run marked four parallel
# "slides" on one strummed chord change at Angie t=1273.1 — when the target
# shares an onset cluster with 3+ notes the soft-attack evidence describes
# the STRUM, not a legato move. <= 2 keeps real double-stop hammers (the
# LBTD E-shape pair at t=273.8 lands both targets in one 2-note cluster).
_MAX_TARGET_CLUSTER_SIZE = 2

# The legato audio gate, all three ANDed (calibrated on the LBTD intro riff
# + full-tempo demo; measured second-onset values):
#   legato-positive: attack 0.12-0.81, percentile 0.00-0.20, ratio 0.12-0.86
#   picked candidates: attack >= 0.88, percentile >= 0.26, ratio >= 0.90
#     (closest negatives: 26→29 attack 0.884/pctl 0.26/ratio 0.99,
#      11→12 0.911/0.31/1.03)
# The absolute cut alone does not transfer to fingerstyle material (Angie
# picked notes measure 0.2-0.6), which is what the percentile term fixes;
# the ratio term protects sparse windows where the percentile is noisy.
_ATTACK_MAX_ABS = 0.85
_ATTACK_MAX_PCTL = 0.25
_ATTACK_MAX_RATIO = 0.90

# Minimum onsets (one per cluster) in a window for the percentile to mean
# anything. Below this the gate fails closed (no detection) — conservative.
_MIN_PCTL_POPULATION = 8

# ---------------------------------------------------------------------------
# Bend gates
# ---------------------------------------------------------------------------

# Second onset must start inside the struck note's sustain. MT3 end times
# run slightly short (the Angie 1451 s member starts 0.06 s after the struck
# note's recorded end), hence the tolerance past ``end``.
#
# The minimum gap EQUALS _MAX_LEGATO_GAP and the boundary belongs to the
# LEGATO side (the bend gate rejects gap <= this, the legato gate accepts
# gap <= it), making the candidate ranges truly disjoint: an instant
# hammer step is smeared into an
# apparent glide by the high-res CQT's filter length (~0.2 s at MIDI 51-58),
# so the pre-glide gate alone cannot always separate the two, and the first
# runs typed LBTD's labeled D|1h2 hammer (gap 0.35 s) as a bend. Pairs
# arriving on a rhythmic subdivision (<= 0.45 s) route to the legato
# detector; only the hold-THEN-bend pattern (Angie flagship: member at
# +0.54 s) stays a bend candidate. Fast strike-and-bend licks are missed —
# accepted recall cost, the mark would otherwise be a coin flip vs hammer.
_BEND_MIN_ONSET_GAP = _MAX_LEGATO_GAP
_BEND_SUSTAIN_TOL = 0.25

# Mid-flight/target members carry no pick attack, but they often share an
# onset cluster with re-struck top strings, so the gate is looser than the
# legato percentile (the trajectory below is the strong gate). Angie's
# 1451.55 s member measures attack 0.295 vs struck 0.456.
_BEND_ATTACK_MAX_PCTL = 0.50

# High-resolution pitch-track parameters: 3 bins/semitone, hop 256
# (~11.6 ms). The band spans [p0-1, p0+2.67]; frames whose argmax sits
# within _BEND_EDGE_GUARD semitones of either band edge are masked as
# leakage from out-of-band neighbors (the Angie case's louder C#4 cluster-
# mate one bin above the band, the open-e above the false 61→62 candidate).
_BEND_BPS = 3
_BEND_HOP = 256
_BEND_BAND_BELOW = 1.0
_BEND_N_BINS = 12  # p0-1 .. p0+8/3 at 3 bins/semitone
_BEND_EDGE_GUARD = 0.45

# Track spans: early frames establish the struck pitch, tail frames the
# arrival pitch. Angie 1451 s measures early 58.02 (|Δ| 0.02 <= 0.35) and
# tail 59.04 (rise 1.02 >= 0.75 → target 1).
_BEND_EARLY_SPAN = (0.02, 0.18)
_BEND_TAIL_SPAN = 0.25
_BEND_MIN_FRAMES = 3
_BEND_MAX_SOURCE_DEV = 0.35
_BEND_MIN_RISE = 0.75
_BEND_MAX_MEMBER_SUSTAIN = 1.5  # cap tracked tail length (s)

# Bend-vs-hammer disambiguation. A fretted hammer-on produces the same
# no-attack pitch STEP as a bend's arrival, so the tail alone cannot tell
# them apart (the first run typed LBTD's labeled D|1h2 hammer as a bend).
# What separates them is the glide: a bent pitch is already BETWEEN the
# fret pitches just before the member note's transcribed onset (MT3 lags
# the glide — Angie's 1451 s bend reads 58.93 in the pre-onset frames),
# while a hammer sits exactly at the source pitch until it lands (the
# LBTD 1h2 pre-onset frames read 51.0). The LAST unmasked frame in the
# _BEND_PREGLIDE_SPAN seconds before the member onset must already sit
# _BEND_MIN_PREGLIDE semitones above the struck pitch — last, not median,
# because a fast bend's glide spans only a few frames (the Angie flagship
# glide takes ~50 ms: pre-window reads 58.03, 58.03, 4×masked, 58.91).
_BEND_PREGLIDE_SPAN = (0.09, 0.005)
_BEND_MIN_PREGLIDE = 0.4

# Frames quieter than this fraction of the band's loudest frame are masked
# (decayed tails would otherwise contribute noise pitches).
_BEND_MIN_LEVEL = 0.05

# ---------------------------------------------------------------------------
# Harmonic gates
# ---------------------------------------------------------------------------

# Conditions (a)+(b): 12th-fret node pitch, long and isolated. The labeled
# Angie pickup harmonic is 1.6 s; ordinary melody notes rarely sustain 0.8 s.
_HARMONIC_NODE_FRET = 12
_HARMONIC_MIN_DURATION = 0.8

# Sustain window over which the spectral profile is measured — past the
# attack, before neighboring notes bleed in.
_HARMONIC_SUSTAIN_START = 0.20
_HARMONIC_SUSTAIN_END = 0.70

# Spectral gate, calibrated on Angie (87 long singleton open+12 candidates;
# the labeled t=1.0 s harmonic is the ONLY one passing all four cuts):
#                     labeled    runner-up   candidate distribution
#   purity E(p)/Σcol   0.307       0.185      med 0.060 / p75 0.092
#   E(p+12)/E(p)       0.176         —        med 0.906
#   E(p+19)/E(p)       0.099         —        med 0.800
#   E(p-12)/E(p)       0.011         —        med 0.067
# A touched 12th-fret harmonic rings nearly pure: upper partials are weak
# and the open-string fundamental (p-12) is suppressed by the touch. The
# LBTD fretted-A3 negatives measure purity 0.23, r12 0.33-0.37, r19
# 0.23-0.24 — rejected on three of four cuts.
_HARMONIC_MIN_PURITY = 0.25
_HARMONIC_MAX_R12 = 0.25
_HARMONIC_MAX_R19 = 0.15
_HARMONIC_MAX_RM12 = 0.05


# ---------------------------------------------------------------------------
# Attack measurement (shared per-window HPSS envelopes)
# ---------------------------------------------------------------------------


class _AttackMeasurer:
    """Per-onset attack strength over shared per-window HPSS envelopes.

    attack(onset) = peak percussive energy >= 2 kHz in [onset-20ms,
    onset+70ms] divided by the mean full-band harmonic energy in
    [onset+30ms, onset+200ms]. One STFT+HPSS per analysis window, cached.
    """

    def __init__(self, stem_path: Path, window_seconds: float):
        self._stem_path = stem_path
        self._window_seconds = window_seconds
        self._envs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cache: dict[float, float | None] = {}

    def window_index(self, onset: float) -> int:
        return int(onset // self._window_seconds)

    def _env(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if k not in self._envs:
            start = k * self._window_seconds
            end = (k + 1) * self._window_seconds + _WINDOW_PAD
            y, sr = load_stem_window(self._stem_path, start, end, sr=_SR)
            S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
            harm, perc = librosa.decompose.hpss(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
            times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=_HOP) + start
            self._envs[k] = (perc[freqs >= _ATTACK_BAND_HZ, :].sum(axis=0), harm.sum(axis=0), times)
        return self._envs[k]

    def attack(self, onset: float) -> float | None:
        if onset < 0:
            return None
        if onset in self._cache:
            return self._cache[onset]
        perc_hi, harm, times = self._env(self.window_index(onset))
        m_attack = (times >= onset - _ATTACK_PRE) & (times <= onset + _ATTACK_POST)
        m_harm = (times >= onset + _ATTACK_HARM_START) & (times <= onset + _ATTACK_HARM_END)
        value: float | None = None
        if m_attack.any() and m_harm.any():
            h = float(harm[m_harm].mean())
            if h > 0.0:
                value = float(perc_hi[m_attack].max()) / h
        self._cache[onset] = value
        return value


def _window_populations(records: list[dict], measurer: _AttackMeasurer) -> dict[int, list[float]]:
    """Attack values for each window's cluster onsets (one per cluster) —
    the population the percentile gate ranks candidates against."""
    first_onset: dict[int, float] = {}
    for n in records:  # records are start-sorted
        first_onset.setdefault(n["cluster_id"], n["start"])
    populations: dict[int, list[float]] = defaultdict(list)
    for onset in first_onset.values():
        a = measurer.attack(onset)
        if a is not None:
            populations[measurer.window_index(onset)].append(a)
    return populations


def _attack_percentile(
    value: float, onset: float, measurer: _AttackMeasurer, populations: dict[int, list[float]]
) -> float | None:
    pop = populations.get(measurer.window_index(onset), [])
    if len(pop) < _MIN_PCTL_POPULATION:
        return None
    return float((np.asarray(pop) < value).mean())


# ---------------------------------------------------------------------------
# Hammer / pull / slide
# ---------------------------------------------------------------------------


def _legato_type(from_fret: int, to_fret: int) -> str | None:
    """Classify a same-string fret move, or None when it's no candidate."""
    delta = to_fret - from_fret
    if delta == 0:
        return None
    if from_fret == 0 and 1 <= delta <= _OPEN_LEGATO_MAX_DELTA:
        return "hammer"  # can't slide from an open string (A|0h3 figures)
    if to_fret == 0 and 1 <= -delta <= _OPEN_LEGATO_MAX_DELTA:
        return "pull"  # can't slide to an open string (e|3p0 figures)
    if from_fret == 0 or to_fret == 0:
        return None  # wider open-endpoint moves are position changes
    if abs(delta) <= _HP_MAX_DELTA:
        return "hammer" if delta > 0 else "pull"
    if _SLIDE_MIN_DELTA <= abs(delta) <= _SLIDE_MAX_DELTA:
        return "slide"
    return None


def _detect_hammer_pull_slides(
    records: list[dict],
    measurer: _AttackMeasurer,
    populations: dict[int, list[float]],
    excluded: set[int],
) -> list[dict]:
    by_string: dict[int, list[dict]] = defaultdict(list)
    for n in records:
        by_string[n["string"]].append(n)
    cluster_sizes = Counter(n["cluster_id"] for n in records)

    out: list[dict] = []
    for string_notes in by_string.values():
        for a, b in zip(string_notes, string_notes[1:], strict=False):
            if a["note_index"] in excluded or b["note_index"] in excluded:
                continue
            if cluster_sizes[b["cluster_id"]] > _MAX_TARGET_CLUSTER_SIZE:
                continue
            gap = b["start"] - a["start"]
            if not (0.0 < gap <= _MAX_LEGATO_GAP):
                continue
            kind = _legato_type(a["fret"], b["fret"])
            if kind is None:
                continue
            attack_a = measurer.attack(a["start"])
            attack_b = measurer.attack(b["start"])
            if attack_a is None or attack_b is None or attack_a <= 0.0:
                continue
            ratio = attack_b / attack_a
            pctl = _attack_percentile(attack_b, b["start"], measurer, populations)
            if pctl is None:
                continue
            if attack_b > _ATTACK_MAX_ABS or pctl > _ATTACK_MAX_PCTL or ratio > _ATTACK_MAX_RATIO:
                continue
            out.append(
                {
                    "type": kind,
                    "from_note_index": a["note_index"],
                    "note_index": b["note_index"],
                    "string": b["string"],
                    "from_fret": a["fret"],
                    "to_fret": b["fret"],
                    "evidence": {
                        "onset_ratio": round(ratio, 3),
                        "attack": round(attack_b, 3),
                        "attack_pctl": round(pctl, 3),
                        "first_attack": round(attack_a, 3),
                        "gap": round(gap, 3),
                    },
                }
            )
    return out


# ---------------------------------------------------------------------------
# Bends
# ---------------------------------------------------------------------------


def _bend_pitch_track(
    stem_path: Path, p0: int, t0: float, t1: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """High-res (3 bins/semitone) pitch track in the bend band over
    [t0, t1]. Returns (times, pitches) with masked frames as NaN — masked =
    near a band edge (leakage from louder out-of-band neighbors) or quieter
    than _BEND_MIN_LEVEL of the band's loudest frame."""
    load_start = max(0.0, t0 - 0.3)
    y, sr = load_stem_window(stem_path, load_start, t1 + 0.3, sr=_SR)
    if len(y) == 0:
        return None
    lo_midi = p0 - _BEND_BAND_BELOW
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=_BEND_HOP,
            fmin=librosa.midi_to_hz(lo_midi),
            n_bins=_BEND_N_BINS,
            bins_per_octave=12 * _BEND_BPS,
        )
    )
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=_BEND_HOP) + load_start
    span = (times >= t0) & (times <= t1)
    if not span.any():
        return None
    C, times = C[:, span], times[span]
    level_floor = _BEND_MIN_LEVEL * float(C.max())
    band_lo = lo_midi + _BEND_EDGE_GUARD
    band_hi = lo_midi + (_BEND_N_BINS - 1) / _BEND_BPS - _BEND_EDGE_GUARD
    pitches = np.full(C.shape[1], np.nan)
    for i in range(C.shape[1]):
        col = C[:, i]
        if float(col.max()) < level_floor:
            continue
        j = int(col.argmax())
        offset = 0.0
        if 0 < j < len(col) - 1:
            denom = col[j - 1] - 2.0 * col[j] + col[j + 1]
            if denom != 0.0:
                offset = float(0.5 * (col[j - 1] - col[j + 1]) / denom)
        pitch = lo_midi + (j + offset) / _BEND_BPS
        if band_lo <= pitch <= band_hi:
            pitches[i] = pitch
    return times, pitches


def _measure_bend(stem_path: Path, struck: dict, member: dict) -> dict | None:
    """Trajectory evidence for one bend candidate, or None when the track
    does not show the struck pitch GLIDING up to a sustained tail."""
    p0 = struck["pitch"]
    t0 = struck["start"]
    tb = member["start"]
    t1 = min(member["end"], member["start"] + _BEND_MAX_MEMBER_SUSTAIN)
    track = _bend_pitch_track(stem_path, p0, t0, t1)
    if track is None:
        return None
    times, pitches = track

    early = pitches[(times >= t0 + _BEND_EARLY_SPAN[0]) & (times <= t0 + _BEND_EARLY_SPAN[1])]
    pre = pitches[(times >= tb - _BEND_PREGLIDE_SPAN[0]) & (times <= tb - _BEND_PREGLIDE_SPAN[1])]
    tail = pitches[(times >= t1 - _BEND_TAIL_SPAN) & (times <= t1)]
    early = early[~np.isnan(early)]
    pre_valid = pre[~np.isnan(pre)]
    tail_valid = tail[~np.isnan(tail)]
    if len(early) < _BEND_MIN_FRAMES or len(tail_valid) < _BEND_MIN_FRAMES:
        return None
    early_med = float(np.median(early))
    tail_med = float(np.median(tail_valid))
    if abs(early_med - p0) > _BEND_MAX_SOURCE_DEV:
        return None
    rise = tail_med - early_med
    if rise < _BEND_MIN_RISE:
        return None
    # Glide gate (bend-vs-hammer): the pitch must already be in flight
    # before the member's onset. No observable pre-onset frames → reject
    # (precision-first: an unobservable glide cannot support a bend mark).
    if len(pre_valid) == 0:
        return None
    pre_pitch = float(pre_valid[-1])  # last observable frame before onset
    if pre_pitch - p0 < _BEND_MIN_PREGLIDE:
        return None
    return {
        "target_semitones": int(np.clip(round(rise), 1, 2)),
        "source_pitch_track": round(early_med, 2),
        "preglide_pitch_track": round(pre_pitch, 2),
        "tail_pitch_track": round(tail_med, 2),
        "tail_frames": int(len(tail_valid)),
        # Dedup quality: how much of the tail the track actually saw —
        # false candidates under louder neighbors lose tail frames to the
        # edge-guard masking.
        "tail_coverage": round(float(len(tail_valid)) / max(1, len(tail)), 3),
    }


def _detect_bends(
    records: list[dict],
    stem_path: Path,
    measurer: _AttackMeasurer,
    populations: dict[int, list[float]],
    excluded: set[int],
) -> list[dict]:
    by_string: dict[int, list[dict]] = defaultdict(list)
    for n in records:
        by_string[n["string"]].append(n)

    candidates: list[dict] = []
    for string_notes in by_string.values():
        for a, b in zip(string_notes, string_notes[1:], strict=False):
            if a["note_index"] in excluded or b["note_index"] in excluded:
                continue  # harmonic notes are spoken for entirely
            if a["fret"] < 1:
                continue  # an open string cannot be bent
            if b["pitch"] - a["pitch"] not in (1, 2):
                continue
            gap = b["start"] - a["start"]
            if gap <= _BEND_MIN_ONSET_GAP or b["start"] > a["end"] + _BEND_SUSTAIN_TOL:
                continue
            attack_b = measurer.attack(b["start"])
            if attack_b is None:
                continue
            pctl = _attack_percentile(attack_b, b["start"], measurer, populations)
            if pctl is None or pctl > _BEND_ATTACK_MAX_PCTL:
                continue
            traj = _measure_bend(stem_path, a, b)
            if traj is None:
                continue
            candidates.append(
                {
                    "type": "bend",
                    "note_index": a["note_index"],
                    "string": a["string"],
                    "fret": a["fret"],
                    "target_semitones": traj["target_semitones"],
                    "member_note_indices": [b["note_index"]],
                    "evidence": {
                        "member_attack": round(attack_b, 3),
                        "member_attack_pctl": round(pctl, 3),
                        "source_pitch_track": traj["source_pitch_track"],
                        "preglide_pitch_track": traj["preglide_pitch_track"],
                        "tail_pitch_track": traj["tail_pitch_track"],
                        "tail_coverage": traj["tail_coverage"],
                    },
                    "_quality": traj["tail_coverage"] * traj["tail_frames"],
                    "_cluster_pair": (a["cluster_id"], b["cluster_id"]),
                }
            )

    # One bend per (struck-cluster, member-cluster) pair: simultaneous
    # cluster-mates of a real bend (the Angie C#4→D4 false candidate one
    # string over) can produce a parallel candidate; keep the trajectory
    # with the best-observed tail.
    best_by_pair: dict[tuple[int, int], dict] = {}
    for c in candidates:
        key = c["_cluster_pair"]
        if key not in best_by_pair or c["_quality"] > best_by_pair[key]["_quality"]:
            best_by_pair[key] = c

    # Bends must not share notes with each other either: a bend-release-
    # rebend lick chains candidates (a,b) then (b,c), but b is the FIRST
    # bend's hidden arrival, not a new strike — emitting (b,c) would mark a
    # note another entry orders hidden. Greedy claim in struck-onset order
    # (the real strike comes first); ties go to the better-observed tail.
    start_by_index = {n["note_index"]: n["start"] for n in records}
    claimed: set[int] = set()
    out = []
    for c in sorted(
        best_by_pair.values(),
        key=lambda c: (start_by_index[c["note_index"]], -c["_quality"]),
    ):
        touched = {c["note_index"], *c["member_note_indices"]}
        if touched & claimed:
            continue
        claimed |= touched
        c.pop("_quality")
        c.pop("_cluster_pair")
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Natural harmonics
# ---------------------------------------------------------------------------


def _detect_harmonics(
    records: list[dict],
    stem_path: Path,
    tuning: Sequence[int],
    contexts: dict[int, CQTContext] | None,
    window_seconds: float,
) -> list[dict]:
    cluster_sizes = Counter(n["cluster_id"] for n in records)
    node_pitch_to_string = {int(open_pitch) + 12: s for s, open_pitch in enumerate(tuning)}

    local_contexts: dict[int, CQTContext] = {}

    def context_for(onset: float) -> CQTContext:
        k = int(onset // window_seconds)
        if contexts is not None and k in contexts:
            return contexts[k]
        if k not in local_contexts:
            start = k * window_seconds
            y, sr = load_stem_window(stem_path, start, (k + 1) * window_seconds + _WINDOW_PAD)
            local_contexts[k] = compute_cqt_context(y, sr, start)
        return local_contexts[k]

    out: list[dict] = []
    for n in records:
        duration = n["end"] - n["start"]
        if duration < _HARMONIC_MIN_DURATION:
            continue
        if cluster_sizes[n["cluster_id"]] > 1:
            continue
        open_string = node_pitch_to_string.get(n["pitch"])
        if open_string is None:
            continue
        ctx = context_for(n["start"])
        bins = {off: ctx.bin_for_pitch(n["pitch"] + off) for off in (0, 12, 19, -12)}
        if any(b is None for b in bins.values()):
            continue  # partial out of CQT range — can't verify, stay silent
        mask = (ctx.frame_times >= n["start"] + _HARMONIC_SUSTAIN_START) & (
            ctx.frame_times <= n["start"] + _HARMONIC_SUSTAIN_END
        )
        if not mask.any():
            continue
        col = ctx.magnitudes[:, mask].mean(axis=1)
        fundamental = float(col[bins[0]])
        total = float(col.sum())
        if fundamental <= 0.0 or total <= 0.0:
            continue
        purity = fundamental / total
        r12 = float(col[bins[12]]) / fundamental
        r19 = float(col[bins[19]]) / fundamental
        rm12 = float(col[bins[-12]]) / fundamental
        if (
            purity < _HARMONIC_MIN_PURITY
            or r12 > _HARMONIC_MAX_R12
            or r19 > _HARMONIC_MAX_R19
            or rm12 > _HARMONIC_MAX_RM12
        ):
            continue
        out.append(
            {
                "type": "harmonic",
                "note_index": n["note_index"],
                "open_string": open_string,
                "node_fret": _HARMONIC_NODE_FRET,
                "evidence": {
                    "purity": round(purity, 3),
                    "partial_ratio_12": round(r12, 3),
                    "partial_ratio_19": round(r19, 3),
                    "open_fundamental_ratio": round(rm12, 3),
                    "duration": round(duration, 3),
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_articulations(
    note_records: list[dict],
    stem_path: str | Path,
    tuning: Sequence[int],
    contexts: dict[int, CQTContext] | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> list[dict]:
    """Detect articulations over frets.json note records.

    ``note_records`` need the fields note_index/start/end/pitch/string/fret/
    cluster_id (exactly what assign_frets emits). ``contexts`` are fret.py's
    per-window CQT contexts (reused by the harmonic detector so the CQT runs
    once); when absent they are computed here for the needed windows only.
    Returns the contract-shaped list, sorted by the primary note's onset —
    empty when nothing passes the gates (callers omit the key then).
    """
    stem_path = Path(stem_path)
    records = sorted(note_records, key=lambda n: (n["start"], n["pitch"]))
    if not records:
        return []

    measurer = _AttackMeasurer(stem_path, window_seconds)
    populations = _window_populations(records, measurer)

    harmonics = _detect_harmonics(records, stem_path, tuning, contexts, window_seconds)
    harmonic_indices = {h["note_index"] for h in harmonics}
    bends = _detect_bends(records, stem_path, measurer, populations, harmonic_indices)

    # Cross-detector disjointness: harmonic notes cannot be bend endpoints
    # (excluded above), bends never share notes with each other (greedy
    # claim inside _detect_bends), and bend members/struck notes and
    # harmonic notes must not double as hammer/pull/slide endpoints — no
    # entry may reference a note another entry orders hidden or re-strung.
    excluded: set[int] = set(harmonic_indices)
    for b in bends:
        excluded.add(b["note_index"])
        excluded.update(b["member_note_indices"])
    legato = _detect_hammer_pull_slides(records, measurer, populations, excluded)

    by_index = {n["note_index"]: n for n in records}
    return sorted(harmonics + bends + legato, key=lambda a: by_index[a["note_index"]]["start"])
