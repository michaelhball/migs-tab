"""Hand evidence for fret assignment: where the fretting fingers are at time t.

Combines ``hand_track.json`` (MediaPipe landmarks per sampled frame) with
``fretboard.json`` (per-frame fretboard geometry) into a snapshot per onset:
for each of the four fingertips, the fret slot the fingertip is behind and
the string it is over. The fret decode adds ``shape_cost`` to every candidate
(string, fret) assignment, so a shape whose fretted notes have no finger near
them is expensive, and it may read a transcribed pitch an octave up or down
when the fingers say so (``octave_shift_penalty``).

``shape_cost`` assigns fretted notes to fingertips jointly: a fingertip
presses one note, the index finger may barre several strings at one fret, and
any other finger may cover adjacent strings at one fret for a small extra
cost. Without that rule one fingertip near fret 4 would make three notes on
three strings look fingered.

Conventions: string index 0 is the low E; ``fret_cont`` is the continuous
fret coordinate from ``FrameGeometry.fret_at`` (1.0 on wire 1, so a finger
pressing fret 2 reads about 1.3-1.9).
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from dataclasses import dataclass

from .fretboard import Fretboard, load_fretboard
from .handtrack import FINGERTIPS
from .paths import VideoPaths

FINGER_NAMES = ("index", "middle", "ring", "pinky")
_TIME_TOL_S = 0.15  # nearest tracked frame must be this close to the onset
_SMOOTH_FRAMES = 3  # frames pooled per snapshot (set to 1 for the nearest frame only)
_SMOOTH_WINDOW_S = 0.25
# Costs (same units as fret.py's shape costs: an open string is worth -0.8,
# a full open-chord template -4.0).
_W_FRET_MISMATCH = 2.5  # per fret beyond the slack
_W_STRING_MISMATCH = 1.0  # per string beyond the slack
_FRET_SLACK = 0.6  # a fingertip within this of the fret slot's centre fits
_STRING_SLACK = 0.7
_CAP_PER_NOTE = 6.0
_HAND_RANGE_SLACK = 1.5  # fretted notes may lie this far outside the fingertip range
_HAND_RANGE_PENALTY = 3.0
OCTAVE_SHIFT_PENALTY = 1.5
_MINI_BARRE_COST = 0.5  # per extra adjacent string one non-index finger covers
_MINI_BARRE_MAX_STRINGS = 3


@dataclass
class Fingertip:
    name: str
    fret_cont: float
    string: int | None
    string_dist: float


@dataclass
class Snapshot:
    t: float  # time of the tracked frame used
    fingertips: list[Fingertip]
    quality: float  # fretboard fit quality for that frame

    @property
    def fret_range(self) -> tuple[float, float]:
        fs = [f.fret_cont for f in self.fingertips]
        return (min(fs), max(fs)) if fs else (0.0, 0.0)

    @property
    def index_fret(self) -> int | None:
        for f in self.fingertips:
            if f.name == "index":
                return int(math.ceil(f.fret_cont - 0.15))
        return None


class HandEvidence:
    def __init__(self, track: dict, fretboard: Fretboard):
        self._fb = fretboard
        self._w = track["frame_width"]
        self._h = track["frame_height"]
        self._frames = [
            f for f in track["frames"] if any(h["role"] == "fretting" for h in f["hands"])
        ]
        self._ts = [f["t"] for f in self._frames]

    @classmethod
    def load(cls, paths: VideoPaths) -> HandEvidence | None:
        if not (paths.hand_track_json.exists() and paths.fretboard_json.exists()):
            return None
        track = json.loads(paths.hand_track_json.read_text())
        return cls(track, load_fretboard(paths))

    def at(self, t: float) -> Snapshot | None:
        """Fingertip positions around ``t``: the nearest tracked frame must be
        within ``_TIME_TOL_S``; up to ``_SMOOTH_FRAMES`` frames within
        ``_SMOOTH_WINDOW_S`` are pooled per finger (median fret, most common
        string) to damp landmark jitter."""
        if not self._frames:
            return None
        i = bisect.bisect_left(self._ts, t)
        cands = [k for k in (i - 1, i) if 0 <= k < len(self._frames)]
        k = min(cands, key=lambda k: abs(self._ts[k] - t))
        if abs(self._ts[k] - t) > _TIME_TOL_S:
            return None
        near = sorted(
            (abs(self._ts[j] - t), j)
            for j in range(
                max(0, k - _SMOOTH_FRAMES), min(len(self._frames), k + _SMOOTH_FRAMES + 1)
            )
            if abs(self._ts[j] - t) <= _SMOOTH_WINDOW_S
        )[:_SMOOTH_FRAMES]
        per_finger: list[list[Fingertip]] = [[] for _ in FINGER_NAMES]
        quality = 0.0
        for _, j in near:
            frame = self._frames[j]
            geom = self._fb.at(frame["t"])
            if geom is None:
                continue
            quality = max(quality, geom.quality)
            hand = next(h for h in frame["hands"] if h["role"] == "fretting")
            for fi, (name, lm) in enumerate(zip(FINGER_NAMES, FINGERTIPS, strict=True)):
                lx, ly = hand["landmarks"][lm]
                x, y = lx * self._w, ly * self._h
                string, dist = geom.string_at(x, y)
                per_finger[fi].append(Fingertip(name, geom.fret_at(x), string, dist))
        if not per_finger[0]:
            return None
        tips: list[Fingertip] = []
        for name, obs in zip(FINGER_NAMES, per_finger, strict=True):
            fret = statistics.median(o.fret_cont for o in obs)
            strings = [o.string for o in obs if o.string is not None]
            string = statistics.mode(strings) if strings else None
            dist = statistics.median(o.string_dist for o in obs)
            tips.append(Fingertip(name, fret, string, dist))
        return Snapshot(t=self._frames[k]["t"], fingertips=tips, quality=quality)


def finger_costs(snap: Snapshot, string: int, fret: int) -> list[float]:
    """Cost of each fingertip being the one that presses (string, fret):
    distance in frets and strings beyond a little slack. The index finger at
    the note's fret covers the strings from its tip toward the treble side
    for free (a barre runs from the tip across the higher strings)."""
    centre = fret - 0.5  # a finger pressing fret f sits mid-slot behind wire f
    out: list[float] = []
    for tip in snap.fingertips:
        d_fret = max(0.0, abs(tip.fret_cont - centre) - _FRET_SLACK)
        if tip.string is None:
            d_string = 2.0
        elif tip.name == "index" and d_fret == 0.0:
            d_string = max(0.0, (tip.string - string) - _STRING_SLACK - 1.0)
        else:
            d_string = max(0.0, abs(tip.string - string) - _STRING_SLACK)
        out.append(_W_FRET_MISMATCH * d_fret + _W_STRING_MISMATCH * d_string)
    return out


def note_cost(snap: Snapshot, string: int, fret: int) -> float:
    """Cost of one fretted note on its own: the cheapest fingertip, capped.
    Open strings cost nothing. A lower bound on the note's share of
    ``joint_cost``, where fingertips cannot be shared freely."""
    if fret <= 0 or not snap.fingertips:
        return 0.0
    return min(_CAP_PER_NOTE, *finger_costs(snap, string, fret))


def joint_cost(snap: Snapshot, combo: tuple[tuple[int, int], ...]) -> float:
    """Cheapest way to give every fretted note of ``combo`` a fingertip.

    Rules: a note left without a finger pays the per-note cap. The index
    finger may hold several notes at one fret (a barre), but then no string
    between the barred strings may be played open. Any other finger holds one
    note, or adjacent strings at one fret at ``_MINI_BARRE_COST`` per extra
    string."""
    fretted = [(s, f) for s, f in combo if f > 0]
    if not fretted or not snap.fingertips:
        return 0.0
    open_strings = {s for s, f in combo if f == 0}
    costs = [finger_costs(snap, s, f) for s, f in fretted]
    is_index = [tip.name == "index" for tip in snap.fingertips]
    n, n_tips = len(fretted), len(snap.fingertips)
    # Cheapest notes last would prune poorly; visit the hardest notes first.
    order = sorted(range(n), key=lambda i: -min(costs[i]))
    floor = [min(_CAP_PER_NOTE, *costs[i]) for i in order]
    remaining = [sum(floor[k:]) for k in range(n + 1)]
    load: list[list[int]] = [[] for _ in range(n_tips)]
    best = [n * _CAP_PER_NOTE + 1.0]  # above the cost of nobody holding anything

    def allowed(t: int, i: int) -> float | None:
        """Extra cost of adding note i to fingertip t, or None when impossible."""
        held = load[t]
        if not held:
            return 0.0
        if fretted[held[0]][1] != fretted[i][1]:
            return None  # one fingertip, one fret
        strings = [fretted[k][0] for k in held] + [fretted[i][0]]
        lo, hi = min(strings), max(strings)
        if is_index[t]:
            return None if any(lo < s < hi for s in open_strings) else 0.0
        if hi - lo > _MINI_BARRE_MAX_STRINGS - 1:
            return None
        return _MINI_BARRE_COST

    def contiguous() -> bool:
        """Every non-index fingertip's notes sit on adjacent strings."""
        for t, held in enumerate(load):
            if is_index[t] or len(held) < 2:
                continue
            strings = [fretted[k][0] for k in held]
            if max(strings) - min(strings) != len(strings) - 1:
                return False
        return True

    def rec(k: int, acc: float) -> None:
        if acc + remaining[k] >= best[0]:
            return
        if k == n:
            if contiguous():
                best[0] = acc
            return
        i = order[k]
        for t in sorted(range(n_tips), key=lambda t: costs[i][t]):
            c = costs[i][t]
            if c >= _CAP_PER_NOTE:
                break
            extra = allowed(t, i)
            if extra is None:
                continue
            load[t].append(i)
            rec(k + 1, acc + c + extra)
            load[t].pop()
        rec(k + 1, acc + _CAP_PER_NOTE)

    rec(0, 0.0)
    return best[0]


def range_penalty(snap: Snapshot, combo: tuple[tuple[int, int], ...]) -> float:
    """Penalty when every fretted note lies outside the fingertips' fret range."""
    fs = [f for _, f in combo if f > 0]
    if not fs or not snap.fingertips:
        return 0.0
    lo, hi = snap.fret_range
    if max(fs) - 0.5 < lo - _HAND_RANGE_SLACK or min(fs) - 0.5 > hi + _HAND_RANGE_SLACK:
        return _HAND_RANGE_PENALTY
    return 0.0


def shape_cost(snap: Snapshot, combo: tuple[tuple[int, int], ...]) -> float:
    """Extra cost of a candidate shape given where the fingers are:
    ``joint_cost`` over its fretted notes plus ``range_penalty``."""
    return joint_cost(snap, combo) + range_penalty(snap, combo)
