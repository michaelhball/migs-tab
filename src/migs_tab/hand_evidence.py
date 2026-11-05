"""Hand evidence for fret assignment: where the fretting fingers are at time t.

Combines ``hand_track.json`` (MediaPipe landmarks per sampled frame) with
``fretboard.json`` (per-frame fretboard geometry) into a snapshot per onset:
for each of the four fingertips, the fret slot the fingertip is behind and
the string it is over. The fret decode adds ``shape_cost`` to every candidate
(string, fret) assignment, so a shape whose fretted notes have no finger near
them is expensive, and it may read a transcribed pitch an octave up or down
when the fingers say so (``octave_shift_penalty``).

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


def note_cost(snap: Snapshot, string: int, fret: int) -> float:
    """Cost of one fretted note given the fingertips: distance in frets and
    strings to the nearest fingertip beyond a little slack, capped. The index
    finger counts as covering every string at its fret (a barre). Open
    strings cost nothing."""
    if fret <= 0 or not snap.fingertips:
        return 0.0
    centre = fret - 0.5  # a finger pressing fret f sits mid-slot behind wire f
    best = _CAP_PER_NOTE
    for tip in snap.fingertips:
        d_fret = max(0.0, abs(tip.fret_cont - centre) - _FRET_SLACK)
        if tip.name == "index" and d_fret == 0.0:
            d_string = 0.0
        elif tip.string is None:
            d_string = 2.0
        else:
            d_string = max(0.0, abs(tip.string - string) - _STRING_SLACK)
        best = min(best, _W_FRET_MISMATCH * d_fret + _W_STRING_MISMATCH * d_string)
    return best


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
    """Extra cost of a candidate shape given where the fingers are: the sum
    of ``note_cost`` over its fretted notes plus ``range_penalty``."""
    return sum(note_cost(snap, s, f) for s, f in combo) + range_penalty(snap, combo)
