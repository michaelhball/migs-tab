"""Tests for fretboard.py — string selection, wire crossings, the anchored
ladder fit, smoothing, and the pixel→fret map."""

from __future__ import annotations

import math

import numpy as np
import pytest

from migs_tab.fretboard import (
    FrameGeometry,
    Fretboard,
    band_crossing_profile,
    crossing_peaks,
    fit_ladder_anchored,
    nut_candidates,
    select_six_strings,
    smooth_geometries,
)


def _model_x(p0: float, p1: float, p2: float, fret: int) -> float:
    u = 1 - 2 ** (-fret / 12)
    return (p0 + p1 * u) / (1 + p2 * u)


def _cluster(yc: float, length: float = 1000.0, ang: float = -1.5, b: float = -0.02) -> dict:
    return {"a": yc - b * 640, "b": b, "ang": ang, "yc": yc, "length": length}


class TestSelectSixStrings:
    def test_picks_the_family_containing_the_hand_over_a_decoy_grid(self):
        strings = [_cluster(373 + 18.5 * i, length=2000) for i in range(6)]
        binding = [_cluster(368, length=7000)]  # fretboard edge just above string 0
        grid = [_cluster(100 + 42 * i, length=800) for i in range(6)]  # chord diagram
        hand = [(1000.0, 420.0), (1030.0, 440.0), (1050.0, 455.0)]
        six = select_six_strings(binding + strings + grid, hand)
        assert [c["yc"] for c in six] == pytest.approx([373 + 18.5 * i for i in range(6)])

    def test_without_hand_prefers_regular_spacing_and_length(self):
        strings = [_cluster(373 + 18.5 * i, length=2000) for i in range(6)]
        strays = [_cluster(404, length=300), _cluster(423, length=300)]
        six = select_six_strings(strings + strays, [])
        assert [c["yc"] for c in six] == pytest.approx([373 + 18.5 * i for i in range(6)])

    def test_none_when_fewer_than_six(self):
        assert select_six_strings([_cluster(300), _cluster(320)], []) is None


class TestCrossings:
    def test_wires_crossing_the_band_are_peaks(self):
        h, w = 200, 400
        th = np.zeros((h, w), dtype=np.uint8)
        strings = [_cluster(60 + 15 * i, b=0.0) for i in range(6)]
        for c in strings:  # the strings themselves are bright rows
            th[int(c["yc"]), :] = 200
        for x in (50, 120, 180, 230):  # full-height wires, 3 px wide like real ones
            th[55:140, x - 1 : x + 2] = 200
        th[90:100, 300] = 200  # a glint that does not cross the band
        prof = band_crossing_profile(th, strings)
        peaks = crossing_peaks(prof)
        xs = sorted(round(x) for x, _ in peaks)
        assert xs == [50, 120, 180, 230]
        assert all(s > 0.8 for _, s in peaks)

    def test_nut_candidates_require_strength_and_a_plausible_gap(self):
        peaks = [(1167.0, 0.93), (1068.0, 0.7), (983.0, 0.7), (1273.0, 0.35), (1204.0, 0.4)]
        assert nut_candidates(peaks, True, 1280) == [1167.0]
        # A strong peak with no strong neighbour at a fret-1 distance is not a nut.
        assert nut_candidates([(1167.0, 0.93), (300.0, 0.9)], True, 1280) == []


class TestAnchoredFit:
    def test_recovers_perspective_ladder_with_fingers_and_strays(self):
        p0, p1, p2 = 1167.0, -1500.0, 0.3
        peaks = [(_model_x(p0, p1, p2, f), 0.8) for f in [0, *range(3, 21)]]
        peaks += [(983.0, 0.45), (1040.0, 0.5)]  # fingers at frets 1-2
        peaks += [(1204.0, 0.4), (1273.0, 0.35)]  # tuner glints beyond the nut
        peaks += [(472.0, 0.32), (549.0, 0.33)]  # inlays
        fit = fit_ladder_anchored(peaks, p0, True, 1280)
        assert fit is not None
        (q0, q1, q2), diag = fit
        assert q0 == pytest.approx(p0, abs=4)  # refinement may move the nut a few px
        assert q2 == pytest.approx(p2, abs=0.08)
        assert _model_x(q0, q1, q2, 12) == pytest.approx(_model_x(p0, p1, p2, 12), abs=3)
        assert diag["quality"] >= 0.75
        assert 0 in diag["matched"] and 12 in diag["matched"]

    def test_mirrored(self):
        p0, p1, p2 = 100.0, 1500.0, 0.0
        peaks = [(_model_x(p0, p1, p2, f), 0.8) for f in range(0, 19)]
        fit = fit_ladder_anchored(peaks, p0, False, 1280)
        assert fit is not None
        (q0, q1, _), diag = fit
        assert q1 == pytest.approx(p1, abs=30)
        assert diag["quality"] == 1.0

    def test_none_on_too_few(self):
        assert fit_ladder_anchored([(10.0, 1.0)], 10.0, True, 1280) is None


def _geom(t: float, p0=1200.0, p1=-1400.0, p2=0.0, quality=1.0) -> FrameGeometry:
    return FrameGeometry(
        t=t,
        strings=[[397.0 + 18.0 * i, 0.0] for i in range(6)],
        p0=p0,
        p1=p1,
        p2=p2,
        quality=quality,
        rms=1.0,
        nut_strength=0.9,
    )


class TestFrameGeometry:
    @pytest.mark.parametrize("p2", [0.0, 0.3, -0.2])
    def test_fret_at_is_exact_on_wires(self, p2):
        g = _geom(0.0, p2=p2)
        for f in (1, 5, 12):
            assert g.fret_at(g.wire_x(f)) == pytest.approx(f, abs=1e-6)

    def test_pressed_fret_is_the_wire_toward_the_body(self):
        g = _geom(0.0)
        mid = (g.wire_x(4) + g.wire_x(5)) / 2
        assert g.pressed_fret(mid) == 5
        assert g.pressed_fret(g.wire_x(5) + 2) == 5

    def test_fret_at_clamps_beyond_the_nut(self):
        g = _geom(0.0)
        assert g.fret_at(g.nut_x + 50) == 0.0

    def test_scale_px_matches_linear_case(self):
        assert _geom(0.0).scale_px == pytest.approx(1400.0)

    def test_string_at_picks_nearest_line(self):
        g = _geom(0.0)
        idx, dist = g.string_at(900.0, 436.0)
        assert idx == 2 and dist < 0.2
        idx, dist = g.string_at(900.0, 600.0)
        assert idx == 5 and dist > 2


class TestFretboard:
    def test_at_picks_nearest_frame_within_gap(self):
        fb = Fretboard(1280, 720, True, 2.0, [_geom(0.0), _geom(0.5), _geom(10.0)])
        assert fb.at(0.6).t == 0.5
        assert fb.at(9.0).t == 10.0
        assert fb.at(5.0) is None  # 4.5 s from the nearest frame

    def test_locate_reports_fret_and_string(self):
        fb = Fretboard(1280, 720, True, 2.0, [_geom(0.0)])
        g = fb.frames[0]
        x = (g.wire_x(2) + g.wire_x(3)) / 2
        loc = fb.locate(0.1, x / 1280, 433.0 / 720)
        assert loc["fret"] == 3 and loc["string"] == 2

    def test_roundtrip_json(self):
        fb = Fretboard(1280, 720, True, 2.0, [_geom(0.0, p2=0.25), _geom(0.5)], {"k": 1})
        again = Fretboard.from_json(fb.to_json())
        assert again.frames[0].p2 == 0.25 and again.stats == {"k": 1}
        assert math.isclose(again.frames[0].fret_at(1000.0), fb.frames[0].fret_at(1000.0))


class TestSmoothing:
    def test_outlier_frame_is_pulled_to_its_neighbours_and_gaps_borrow(self):
        frames = [_geom(0.0), _geom(0.5), _geom(1.0, p0=1260.0), _geom(1.5), _geom(2.0)]
        frames.append(_geom(2.5, quality=0.0))  # no fit in this frame
        frames.append(_geom(9.0, quality=0.0))  # too far from any good frame
        sm = smooth_geometries(frames, window=5)
        by_t = {g.t: g for g in sm}
        assert by_t[1.0].p0 == pytest.approx(1200.0, abs=1)
        assert by_t[2.5].borrowed_from == 2.0
        assert 9.0 not in by_t
