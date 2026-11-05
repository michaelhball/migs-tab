"""Fretboard geometry per frame: where every fret wire and string sits.

The player shifts the guitar during a tutorial, so the geometry is found per
sampled frame and smoothed over time rather than calibrated once. The map is
what turns the hand landmarks in ``hand_track.json`` into evidence for
string/fret assignment.

Method, per frame
-----------------
1. Top-hat filter (pixel minus local background) so thin bright features
   survive regardless of large bright areas such as the guitar body.
2. Strings: probabilistic Hough on the thresholded top-hat image gives long
   near-horizontal segments; cluster them into lines and pick the six lines
   with the most regular spacing. The fretting hand's fingertips (from the
   hand track) must lie inside the chosen band, which rejects on-screen
   notation staves and chord-diagram grids.
3. Wires: a fret wire is a bright column that crosses the whole string band;
   a stray glint or an inlay does not. Score each column by the fraction of
   band pixels (string rows excluded) that are bright; local maxima are wire
   candidates with that fraction as their strength.
4. Nut: the strongest crossing at the headstock end of the run (bone or
   plastic, wider and brighter than a wire).
5. Ladder: fit x(u) = (p0 + p1 u) / (1 + p2 u) with p0 anchored at the nut,
   where u_n = 1 - 2^(-n/12) is the fraction of the scale length at wire n
   and p2 absorbs the camera's perspective. Hypotheses come from each wire
   candidate taken as fret 1..14 over a small grid of p2; the hypothesis
   explaining the most crossing strength wins, with penalties for strong
   unexplained columns and for model wires that have no candidate.

Then the per-frame ladders are median-smoothed over neighbouring frames and
frames without a fit borrow the nearest good one.

``Fretboard.locate(t, x, y)`` answers the two questions the fret decode asks
about a fingertip: which fret is it behind, and which string is it over.
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .handtrack import FINGERTIPS, iter_frames, probe_video_size
from .paths import VideoPaths

_MAX_FRETS = 20
_U = 1 - 2 ** (-np.arange(0, _MAX_FRETS + 1) / 12)  # scale fraction at wire n

DEFAULT_GEOMETRY_FPS = 2.0
# Top-hat kernel (pixels) and brightness threshold on the top-hat image.
_TOPHAT_KERNEL = 15
_TOPHAT_LINE_THRESHOLD = 28
_TOPHAT_BRIGHT = 22
# Hough strings.
_HOUGH_THRESHOLD = 60
_HOUGH_MIN_LENGTH = 150
_HOUGH_MAX_GAP = 20
_STRING_MAX_ANGLE_DEG = 30.0
_CLUSTER_Y_TOL = 4.0
_CLUSTER_ANGLE_TOL = 2.5
_STRING_MIN_CLUSTER_LEN = 250.0
_STRING_CANDIDATE_CLUSTERS = 14
_STRING_MIN_SPACING = 5.0
_STRING_HAND_REACH_PX = 140.0  # a string line passes within this of the fingertips' centroid
# Wires.
_CROSSING_MIN_STRENGTH = 0.30
_CROSSING_MERGE_PX = 5
_NUT_MIN_STRENGTH = 0.75
_NUT_GAP_RANGE = (0.03, 0.30)  # nut-to-next-visible-wire gap as a fraction of frame width
_STRONG_UNEXPLAINED = 0.65
_MISSING_WIRE_PENALTY = 0.35
_MATCH_TOL_FRACTION = 0.25
_MATCH_TOL_MIN_PX = 3.0
_P2_GRID = (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45)
_MAX_PAIR_FRET = 14
_SCALE_TO_STRING_SPAN = (10.0, 28.0)  # scale_px / (low-E to high-e spacing at the nut)
_MIN_QUALITY = 0.5
# Smoothing.
_SMOOTH_WINDOW = 5
_BORROW_MAX_GAP_S = 3.0


@dataclass
class FrameGeometry:
    t: float
    strings: list[list[float]]  # 6 × [a, b]: y = a + b x; index 0 = low E (top line)
    p0: float
    p1: float
    p2: float
    quality: float  # fraction of visible model wires 0..12 that matched a candidate
    rms: float
    nut_strength: float
    borrowed_from: float | None = None  # t of the frame this geometry was copied from

    # -- geometry -------------------------------------------------------------

    @property
    def nut_x(self) -> float:
        return self.p0

    @property
    def scale_px(self) -> float:
        """Scale length in pixels as seen at the nut (|dx/du| at u = 0)."""
        return abs(self.p1 - self.p0 * self.p2)

    def wire_x(self, fret: int) -> float:
        u = float(_U[fret]) if fret <= _MAX_FRETS else 1 - 2 ** (-fret / 12)
        return (self.p0 + self.p1 * u) / (1 + self.p2 * u)

    def fret_at(self, x: float) -> float:
        """Continuous fret coordinate: 1.0 is on wire 1; 1.5 is halfway between
        wires 1 and 2 (where a finger pressing fret 2 sits)."""
        denom = self.p1 - self.p2 * x
        if abs(denom) < 1e-9:
            return float(_MAX_FRETS)
        u = (x - self.p0) / denom
        if u <= 0:
            return 0.0
        if u >= 1:
            return float(_MAX_FRETS)
        return -12 * math.log2(1 - u)

    def pressed_fret(self, x: float) -> int:
        """Integer fret a fingertip at ``x`` presses: the wire just toward the
        body from the fingertip (fingers sit behind the wire)."""
        return int(math.ceil(self.fret_at(x) - 0.15))

    def string_y(self, index: int, x: float) -> float:
        a, b = self.strings[index]
        return a + b * x

    def string_at(self, x: float, y: float) -> tuple[int | None, float]:
        """Nearest string index to (x, y) and its distance in string spacings."""
        if len(self.strings) != 6:
            return None, math.inf
        ys = [self.string_y(i, x) for i in range(6)]
        spacing = (ys[-1] - ys[0]) / 5
        if spacing <= 0:
            return None, math.inf
        idx = min(range(6), key=lambda i: abs(ys[i] - y))
        return idx, abs(ys[idx] - y) / spacing

    def string_span_at_nut(self) -> float:
        return abs(self.string_y(5, self.p0) - self.string_y(0, self.p0))


@dataclass
class Fretboard:
    frame_width: int
    frame_height: int
    headstock_right: bool
    fps: float
    frames: list[FrameGeometry]
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames.sort(key=lambda g: g.t)
        self._ts = [g.t for g in self.frames]

    def at(self, t: float) -> FrameGeometry | None:
        """Geometry of the sampled frame nearest ``t`` (within the borrow gap);
        None when that frame has no usable fit."""
        if not self.frames:
            return None
        i = bisect.bisect_left(self._ts, t)
        cands = [k for k in (i - 1, i) if 0 <= k < len(self.frames)]
        k = min(cands, key=lambda k: abs(self._ts[k] - t))
        if abs(self._ts[k] - t) > _BORROW_MAX_GAP_S:
            return None
        g = self.frames[k]
        if g.quality < _MIN_QUALITY or len(g.strings) != 6:
            return None
        return g

    def locate(self, t: float, x_norm: float, y_norm: float) -> dict | None:
        """Fret and string under a normalized landmark at time ``t``."""
        g = self.at(t)
        if g is None:
            return None
        x = x_norm * self.frame_width
        y = y_norm * self.frame_height
        string, dist = g.string_at(x, y)
        return {
            "t_geometry": g.t,
            "x": round(x, 1),
            "y": round(y, 1),
            "fret_cont": round(g.fret_at(x), 2),
            "fret": g.pressed_fret(x),
            "string": string,
            "string_dist": round(dist, 2) if math.isfinite(dist) else None,
            "quality": g.quality,
        }

    # -- (de)serialisation ------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "headstock_right": self.headstock_right,
            "fps": self.fps,
            "stats": self.stats,
            "frames": [
                {
                    "t": g.t,
                    "strings": g.strings,
                    "p0": g.p0,
                    "p1": g.p1,
                    "p2": g.p2,
                    "quality": g.quality,
                    "rms": g.rms,
                    "nut_strength": g.nut_strength,
                    "borrowed_from": g.borrowed_from,
                }
                for g in self.frames
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> Fretboard:
        frames = [
            FrameGeometry(
                t=f["t"],
                strings=f["strings"],
                p0=f["p0"],
                p1=f["p1"],
                p2=f["p2"],
                quality=f["quality"],
                rms=f["rms"],
                nut_strength=f["nut_strength"],
                borrowed_from=f.get("borrowed_from"),
            )
            for f in d["frames"]
        ]
        return cls(
            frame_width=d["frame_width"],
            frame_height=d["frame_height"],
            headstock_right=d["headstock_right"],
            fps=d["fps"],
            frames=frames,
            stats=d.get("stats", {}),
        )


# ---------------------------------------------------------------------------
# Per-frame detection
# ---------------------------------------------------------------------------


def tophat(gray: np.ndarray) -> np.ndarray:
    import cv2  # ships with mediapipe; imported here to keep CLI start-up light

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_TOPHAT_KERNEL, _TOPHAT_KERNEL))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def string_clusters(th: np.ndarray) -> list[dict]:
    """Long near-horizontal lines in the top-hat image, clustered into lines.

    Each cluster: a, b (y = a + b x), ang (degrees), length (summed segment
    length), yc (y at the frame's middle column)."""
    import cv2

    h, w = th.shape
    _, mask = cv2.threshold(th, _TOPHAT_LINE_THRESHOLD, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 720,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LENGTH,
        maxLineGap=_HOUGH_MAX_GAP,
    )
    if lines is None:
        return []
    segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(ang) > _STRING_MAX_ANGLE_DEG:
            continue
        segs.append((float(x1), float(y1), float(x2), float(y2), ang, math.hypot(x2 - x1, y2 - y1)))
    clusters: list[dict] = []
    for x1, y1, x2, y2, ang, length in sorted(segs, key=lambda s: -s[5]):
        b = (y2 - y1) / (x2 - x1)
        a = y1 - b * x1
        yc = a + b * w / 2
        for c in clusters:
            if abs(c["yc"] - yc) <= _CLUSTER_Y_TOL and abs(c["ang"] - ang) <= _CLUSTER_ANGLE_TOL:
                c["length"] += length
                break
        else:
            clusters.append({"a": a, "b": b, "ang": ang, "yc": yc, "length": length})
    return sorted(clusters, key=lambda c: c["yc"])


def select_six_strings(
    clusters: list[dict], hand_pts: list[tuple[float, float]]
) -> list[dict] | None:
    """Pick the six clusters that look like strings: regular spacing, shared
    angle, long, and (when known) with the fretting hand's fingertips inside
    the band."""
    cands = [c for c in clusters if c["length"] >= _STRING_MIN_CLUSTER_LEN]
    if hand_pts:
        # Keep only lines passing near the fretting fingertips: drops notation
        # staves and chord grids before the longest-first cut below.
        hx = statistics.mean(x for x, _ in hand_pts)
        hy = statistics.mean(y for _, y in hand_pts)
        cands = [c for c in cands if abs(c["a"] + c["b"] * hx - hy) <= _STRING_HAND_REACH_PX]
    cands = sorted(cands, key=lambda c: -c["length"])[:_STRING_CANDIDATE_CLUSTERS]
    cands = sorted(cands, key=lambda c: c["yc"])
    best: tuple[float, list[dict]] | None = None
    for combo in itertools.combinations(cands, 6):
        ys = [c["yc"] for c in combo]
        gaps = np.diff(ys)
        med = float(np.median(gaps))
        if med < _STRING_MIN_SPACING or gaps.max() > 1.5 * med or gaps.min() < 0.6 * med:
            continue
        angs = [c["ang"] for c in combo]
        if max(angs) - min(angs) > 3.0:
            continue
        cv = float(np.std(gaps) / med)
        inside = 0
        for hx, hy in hand_pts:
            y0 = combo[0]["a"] + combo[0]["b"] * hx
            y5 = combo[5]["a"] + combo[5]["b"] * hx
            if y0 - 1.5 * med <= hy <= y5 + 1.5 * med:
                inside += 1
        if hand_pts and inside == 0:
            continue  # a band the fretting hand is not touching is an overlay or a decoy
        score = sum(c["length"] for c in combo) * (1 + inside) * math.exp(-4 * cv)
        if best is None or score > best[0]:
            best = (score, list(combo))
    return None if best is None else best[1]


def band_crossing_profile(th: np.ndarray, strings: list[dict]) -> np.ndarray:
    """Per column, the fraction of string-band pixels (string rows excluded)
    that are bright: near 1 on a fret wire or the nut, low elsewhere."""
    h, w = th.shape
    xx = np.arange(w)[None, :]
    yy = np.arange(h)[:, None]
    ys = [c["a"] + c["b"] * xx for c in strings]
    med = float(np.median([float(np.mean(ys[i + 1] - ys[i])) for i in range(5)]))
    band = (yy >= ys[0] - 0.4 * med) & (yy <= ys[5] + 0.4 * med)
    on_string = np.zeros((h, w), dtype=bool)
    for y in ys:
        on_string |= np.abs(yy - y) <= 1.5
    valid = band & ~on_string
    bright = (th > _TOPHAT_BRIGHT) & valid
    denom = np.maximum(valid.sum(axis=0), 1)
    return bright.sum(axis=0) / denom


def crossing_peaks(profile: np.ndarray) -> list[tuple[float, float]]:
    """Local maxima of the crossing profile as (x, strength)."""
    sm = np.convolve(profile, np.ones(3) / 3, mode="same")
    n = len(sm)
    peaks = [
        x
        for x in range(3, n - 3)
        if sm[x] >= _CROSSING_MIN_STRENGTH
        and all(sm[x] >= sm[x + k] for k in (-3, -2, -1, 1, 2, 3))
    ]
    merged: list[list[int]] = []
    for x in peaks:
        if merged and x - merged[-1][-1] <= _CROSSING_MERGE_PX:
            merged[-1].append(x)
        else:
            merged.append([x])
    out = []
    for m in merged:
        xc = float(np.mean(m))
        out.append((xc, float(sm[int(round(xc))])))
    return out


def nut_candidates(
    peaks: list[tuple[float, float]],
    headstock_right: bool,
    frame_width: int,
    hand_bound: float | None = None,
) -> list[float]:
    """Strong crossings at the headstock end whose next body-side strong
    neighbour sits a plausible fret-1..3 gap away (the hand may hide wire 1).

    ``hand_bound`` is the headstock-most x the fretting fingertips reach over
    the whole video: fingers never press beyond the nut, so a candidate on the
    body side of that bound (a rosette ring, the saddle) is not the nut."""
    sign = 1.0 if headstock_right else -1.0
    ordered = sorted(peaks, key=lambda p: -sign * p[0])  # headstock side first
    strong = [p for p in ordered if p[1] >= 0.5]
    out: list[float] = []
    lo, hi = _NUT_GAP_RANGE[0] * frame_width, _NUT_GAP_RANGE[1] * frame_width
    for i, (x, s) in enumerate(strong):
        if s < _NUT_MIN_STRENGTH:
            continue
        if hand_bound is not None and sign * (x - hand_bound) < -0.02 * frame_width:
            break  # every later candidate is further onto the body
        nxt = next((q for q in strong[i + 1 :] if sign * (x - q[0]) > 2), None)
        if nxt is None:
            continue
        gap = abs(x - nxt[0])
        if lo <= gap <= hi:
            out.append(x)
        if len(out) >= 3:
            break
    return out


def _ladder_model(p0: float, p1: float, p2: float) -> np.ndarray:
    return (p0 + p1 * _U) / (1 + p2 * _U)


def _match_peaks(xs: np.ndarray, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest candidate within tolerance for each model wire (-1 if none);
    a candidate claimed by two wires keeps the closer."""
    local_gap = np.abs(np.diff(model))
    tol = np.maximum(_MATCH_TOL_MIN_PX, _MATCH_TOL_FRACTION * np.append(local_gap, local_gap[-1]))
    d = np.abs(xs[:, None] - model[None, :])
    nearest = np.argmin(d, axis=0)
    dist = d[nearest, np.arange(model.shape[0])]
    assigned = np.where(dist <= tol, nearest, -1)
    dist = np.where(assigned >= 0, dist, np.inf)
    for p in np.unique(assigned[assigned >= 0]):
        frets = np.flatnonzero(assigned == p)
        if len(frets) > 1:
            keep = frets[np.argmin(dist[frets])]
            for f in frets:
                if f != keep:
                    assigned[f] = -1
                    dist[f] = np.inf
    return assigned, dist


def _score(
    assigned: np.ndarray,
    model: np.ndarray,
    xs: np.ndarray,
    strength: np.ndarray,
    headstock_right: bool,
    frame_width: int,
) -> float:
    sign = 1.0 if headstock_right else -1.0
    score = float(strength[assigned[assigned >= 0]].sum())
    visible = (model >= 0) & (model < frame_width)
    for f in range(1, 13):
        if visible[f] and assigned[f] < 0:
            score -= _MISSING_WIRE_PENALTY
    # Strong candidates between the nut and wire 16 that no wire explains.
    used = set(assigned[assigned >= 0].tolist())
    inner = sign * (model[0] - xs) >= -2
    inner &= sign * (xs - model[16]) >= -2
    for p in np.flatnonzero(inner & (strength >= _STRONG_UNEXPLAINED)):
        if int(p) not in used:
            score -= float(strength[p])
    return score


def fit_ladder_anchored(
    peaks: list[tuple[float, float]], nut_x: float, headstock_right: bool, frame_width: int
) -> tuple[tuple[float, float, float], dict] | None:
    """Fit the projective ladder with the nut fixed at ``nut_x``.

    Returns ((p0, p1, p2), diagnostics) or None when nothing fits."""
    if len(peaks) < 4:
        return None
    sign = 1.0 if headstock_right else -1.0
    xs = np.array([p[0] for p in peaks], dtype=float)
    strength = np.array([p[1] for p in peaks], dtype=float)
    slope_lo, slope_hi = 0.15 * frame_width, 6.0 * frame_width
    best: tuple[float, float, float, float] | None = None
    for b in range(len(xs)):
        if strength[b] < 0.4 or sign * (nut_x - xs[b]) < 10:
            continue
        for j in range(1, _MAX_PAIR_FRET + 1):
            uj = float(_U[j])
            for p2 in _P2_GRID:
                p1 = (xs[b] * (1 + p2 * uj) - nut_x) / uj
                if not (slope_lo <= -sign * (p1 - nut_x * p2) <= slope_hi):
                    continue
                model = _ladder_model(nut_x, p1, p2)
                assigned, _ = _match_peaks(xs, model)
                s = _score(assigned, model, xs, strength, headstock_right, frame_width)
                if best is None or s > best[0]:
                    best = (s, nut_x, p1, p2)
    if best is None:
        return None
    _, p0, p1, p2 = best
    # Refine (p0 within a few px, p1, p2) by weighted least squares:
    # x (1 + p2 u) = p0 + p1 u  ->  [1, u, -x u] · [p0, p1, p2] = x.
    for _ in range(3):
        model = _ladder_model(p0, p1, p2)
        assigned, _ = _match_peaks(xs, model)
        frets = np.flatnonzero(assigned >= 0)
        if len(frets) < 4:
            break
        px = xs[assigned[frets]]
        wgt = np.sqrt(strength[assigned[frets]])
        A = np.column_stack([np.ones_like(px), _U[frets], -px * _U[frets]])
        coef, *_ = np.linalg.lstsq(A * wgt[:, None], px * wgt, rcond=None)
        if abs(coef[2]) >= 0.9 or abs(coef[0] - nut_x) > 6:
            break
        p0, p1, p2 = (float(c) for c in coef)
    model = _ladder_model(p0, p1, p2)
    assigned, dist = _match_peaks(xs, model)
    visible = (model >= 0) & (model < frame_width)
    vis13 = [f for f in range(13) if visible[f]]
    matched13 = [f for f in vis13 if assigned[f] >= 0]
    resid = np.array([dist[f] for f in range(_MAX_FRETS + 1) if assigned[f] >= 0])
    diag = {
        "quality": round(len(matched13) / max(len(vis13), 1), 3),
        "rms": round(float(np.sqrt(np.mean(resid**2))), 2) if resid.size else 0.0,
        "matched": [int(f) for f in np.flatnonzero(assigned >= 0)],
    }
    return (p0, p1, p2), diag


def analyse_frame(
    gray: np.ndarray,
    hand_pts: list[tuple[float, float]],
    headstock_right: bool,
    hand_bound: float | None = None,
) -> tuple[list[list[float]] | None, tuple[float, float, float] | None, dict]:
    """Strings and ladder for one frame. Returns (strings, (p0,p1,p2), diag)."""
    h, w = gray.shape
    th = tophat(gray)
    six = select_six_strings(string_clusters(th), hand_pts)
    if six is None:
        return None, None, {"reason": "no six-string family"}
    strings = [[round(float(c["a"]), 3), round(float(c["b"]), 5)] for c in six]
    peaks = crossing_peaks(band_crossing_profile(th, six))
    nuts = nut_candidates(peaks, headstock_right, w, hand_bound)
    if not nuts:
        return strings, None, {"reason": "no nut candidate", "peaks": len(peaks)}
    best = None
    for nut in nuts:
        fit = fit_ladder_anchored(peaks, nut, headstock_right, w)
        if fit is None:
            continue
        (p0, p1, p2), diag = fit
        g = FrameGeometry(0.0, strings, p0, p1, p2, diag["quality"], diag["rms"], 0.0)
        span = g.string_span_at_nut()
        ratio = g.scale_px / span if span > 0 else 0.0
        if not (_SCALE_TO_STRING_SPAN[0] <= ratio <= _SCALE_TO_STRING_SPAN[1]):
            diag = {**diag, "rejected": f"scale/string-span ratio {ratio:.1f}"}
            continue
        nut_strength = max((s for x, s in peaks if abs(x - nut) <= 3), default=0.0)
        key = (diag["quality"], -diag["rms"])
        if best is None or key > best[0]:
            best = (key, (p0, p1, p2), {**diag, "nut_strength": nut_strength, "nut_x": nut})
    if best is None:
        return strings, None, {"reason": "no ladder fit", "nuts": nuts}
    return strings, best[1], best[2]


# ---------------------------------------------------------------------------
# Whole video
# ---------------------------------------------------------------------------


def _hand_points_lookup(paths: VideoPaths) -> tuple[dict | None, int, int]:
    if not paths.hand_track_json.exists():
        return None, 0, 0
    track = json.loads(paths.hand_track_json.read_text())
    return track, track["frame_width"], track["frame_height"]


def _fingertips_at(track: dict | None, t: float, w: int, h: int) -> list[tuple[float, float]]:
    if track is None or not track["frames"]:
        return []
    i = min(len(track["frames"]) - 1, max(0, int(round(t * track["fps"]))))
    frame = track["frames"][i]
    pts: list[tuple[float, float]] = []
    for hand in frame["hands"]:
        if hand["role"] == "fretting":
            for tip in FINGERTIPS:
                lx, ly = hand["landmarks"][tip]
                pts.append((lx * w, ly * h))
    return pts


def smooth_geometries(
    frames: list[FrameGeometry], window: int = _SMOOTH_WINDOW
) -> list[FrameGeometry]:
    """Median-smooth the ladder over neighbouring good frames.

    The three model wires 0, 7 and 14 are smoothed as positions and the
    projective parameters refitted exactly from them, which keeps p1 and p2
    from drifting apart. String lines are smoothed per coefficient."""
    good = [g for g in frames if g.quality >= _MIN_QUALITY]
    if len(good) < 3:
        return good  # too few fits to smooth or borrow from; never keep unfitted frames
    half = window // 2
    out: list[FrameGeometry] = []
    for i, g in enumerate(good):
        lo, hi = max(0, i - half), min(len(good), i + half + 1)
        nb = good[lo:hi]
        w0 = statistics.median(n.wire_x(0) for n in nb)
        w7 = statistics.median(n.wire_x(7) for n in nb)
        w14 = statistics.median(n.wire_x(14) for n in nb)
        p = _params_from_three_wires(w0, w7, w14)
        strings = [
            [
                round(statistics.median(n.strings[s][0] for n in nb), 3),
                round(statistics.median(n.strings[s][1] for n in nb), 5),
            ]
            for s in range(6)
        ]
        out.append(
            FrameGeometry(
                t=g.t,
                strings=strings,
                p0=round(p[0], 3),
                p1=round(p[1], 3),
                p2=round(p[2], 5),
                quality=g.quality,
                rms=g.rms,
                nut_strength=g.nut_strength,
            )
        )
    # Frames without a fit borrow the nearest smoothed good frame within the gap.
    ts = [g.t for g in out]
    result: list[FrameGeometry] = []
    for g in frames:
        if g.quality >= _MIN_QUALITY:
            result.append(out[bisect.bisect_left(ts, g.t)])
            continue
        i = bisect.bisect_left(ts, g.t)
        cands = [k for k in (i - 1, i) if 0 <= k < len(out)]
        k = min(cands, key=lambda k: abs(ts[k] - g.t))
        if abs(ts[k] - g.t) <= _BORROW_MAX_GAP_S:
            src = out[k]
            result.append(
                FrameGeometry(
                    t=g.t,
                    strings=src.strings,
                    p0=src.p0,
                    p1=src.p1,
                    p2=src.p2,
                    quality=src.quality,
                    rms=src.rms,
                    nut_strength=src.nut_strength,
                    borrowed_from=src.t,
                )
            )
    return result


def _fingertip_reach(track: dict | None, w: int, headstock_right: bool) -> float | None:
    """Headstock-most x (pixels) the fretting fingertips reach over the video
    (97th percentile), the bound the nut must lie beyond."""
    if track is None:
        return None
    xs = [
        hand["landmarks"][tip][0] * w
        for f in track["frames"]
        for hand in f["hands"]
        if hand["role"] == "fretting"
        for tip in FINGERTIPS
    ]
    if len(xs) < 50:
        return None
    return float(np.percentile(xs, 97 if headstock_right else 3))


def _params_from_three_wires(x0: float, x7: float, x14: float) -> tuple[float, float, float]:
    """Exact (p0, p1, p2) through wires 0, 7 and 14: x(1 + p2 u) = p0 + p1 u."""
    u7, u14 = float(_U[7]), float(_U[14])
    A = np.array([[1.0, 0.0, 0.0], [1.0, u7, -x7 * u7], [1.0, u14, -x14 * u14]])
    sol = np.linalg.solve(A, np.array([x0, x7, x14]))
    return float(sol[0]), float(sol[1]), float(sol[2])


def calibrate_fretboard(
    paths: VideoPaths,
    fps: float = DEFAULT_GEOMETRY_FPS,
    headstock_right: bool = True,
    force: bool = False,
) -> Path:
    """Per-frame fretboard geometry for the whole video; writes fretboard.json."""
    out = paths.fretboard_json
    if out.exists() and not force:
        return out
    if not paths.video.exists():
        raise FileNotFoundError(f"video not found at {paths.video}; run download first")
    import cv2

    w, h = probe_video_size(paths.video)
    track, _, _ = _hand_points_lookup(paths)
    hand_bound = _fingertip_reach(track, w, headstock_right)
    t0 = time.monotonic()
    frames: list[FrameGeometry] = []
    reasons: dict[str, int] = {}
    for _i, t, frame in iter_frames(paths.video, fps):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        strings, params, diag = analyse_frame(
            gray, _fingertips_at(track, t, w, h), headstock_right, hand_bound
        )
        if strings is None or params is None:
            reasons[diag.get("reason", "?")] = reasons.get(diag.get("reason", "?"), 0) + 1
            frames.append(
                FrameGeometry(round(t, 3), strings or [], 0.0, 0.0, 0.0, 0.0, math.inf, 0.0)
            )
            continue
        p0, p1, p2 = params
        frames.append(
            FrameGeometry(
                t=round(t, 3),
                strings=strings,
                p0=round(p0, 3),
                p1=round(p1, 3),
                p2=round(p2, 5),
                quality=diag["quality"],
                rms=diag["rms"],
                nut_strength=round(diag["nut_strength"], 3),
            )
        )
    good = [g for g in frames if g.quality >= _MIN_QUALITY]
    smoothed = smooth_geometries(frames)
    fb = Fretboard(
        frame_width=w,
        frame_height=h,
        headstock_right=headstock_right,
        fps=fps,
        frames=smoothed,
        stats={
            "frames_sampled": len(frames),
            "frames_fitted": len(good),
            "frames_borrowed": sum(1 for g in smoothed if g.borrowed_from is not None),
            "frames_without_geometry": len(frames) - len(smoothed),
            "failure_reasons": reasons,
            "quality_median": round(statistics.median(g.quality for g in good), 3) if good else 0.0,
            "rms_median": round(statistics.median(g.rms for g in good), 2) if good else None,
            "nut_x_median": round(statistics.median(g.nut_x for g in good), 1) if good else None,
            "scale_px_median": round(statistics.median(g.scale_px for g in good), 1)
            if good
            else None,
            "elapsed_s": round(time.monotonic() - t0, 1),
        },
    )
    out.write_text(json.dumps(fb.to_json(), separators=(",", ":")))
    return out


def load_fretboard(paths: VideoPaths) -> Fretboard:
    return Fretboard.from_json(json.loads(paths.fretboard_json.read_text()))


# ---------------------------------------------------------------------------
# Debugging
# ---------------------------------------------------------------------------


def frame_at(video: Path, t: float) -> np.ndarray:
    """One RGB frame at time ``t`` (seconds)."""
    import subprocess

    w, h = probe_video_size(video)
    buf = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if len(buf) < w * h * 3:
        raise RuntimeError(f"could not decode a frame at {t:.3f}s from {video}")
    return np.frombuffer(buf[: w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def render_overlay(paths: VideoPaths, t: float, out_path: Path) -> Path:
    """Draw the geometry nearest ``t`` and the tracked fingertips on that frame."""
    import cv2

    fb = load_fretboard(paths)
    g = fb.at(t)
    img = cv2.cvtColor(frame_at(paths.video, t), cv2.COLOR_RGB2BGR)
    if g is None:
        cv2.putText(
            img, f"t={t:.2f}s no geometry", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return out_path
    for s in range(6):
        y0 = int(g.string_y(s, 0))
        y1 = int(g.string_y(s, fb.frame_width - 1))
        cv2.line(img, (0, y0), (fb.frame_width - 1, y1), (255, 128, 0), 1)
    for fret in range(0, _MAX_FRETS + 1):
        x = int(round(g.wire_x(fret)))
        if not 0 <= x < fb.frame_width:
            continue
        ya = int(g.string_y(0, x)) - 18
        yb = int(g.string_y(5, x)) + 18
        colour = (0, 200, 255) if fret in (0, 12) else (0, 255, 0)
        cv2.line(img, (x, ya), (x, yb), colour, 1)
        cv2.putText(img, str(fret), (x - 6, ya - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    label = f"t={t:.2f}s geom t={g.t:.2f}s q={g.quality:.2f} rms={g.rms:.1f}"
    if g.borrowed_from is not None:
        label += f" (borrowed from {g.borrowed_from:.2f}s)"
    if paths.hand_track_json.exists():
        track = json.loads(paths.hand_track_json.read_text())
        frame = min(track["frames"], key=lambda f: abs(f["t"] - t))
        for hand in frame["hands"]:
            if hand["role"] != "fretting":
                continue
            for tip in FINGERTIPS:
                lx, ly = hand["landmarks"][tip]
                loc = fb.locate(t, lx, ly)
                if loc is None:
                    continue
                px, py = int(loc["x"]), int(loc["y"])
                cv2.circle(img, (px, py), 6, (255, 0, 255), 2)
                tag = (
                    f"s{loc['string']}f{loc['fret']}"
                    if loc["string"] is not None
                    else f"f{loc['fret']}"
                )
                cv2.putText(
                    img, tag, (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1
                )
    cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path
