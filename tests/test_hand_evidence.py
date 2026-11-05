"""Tests for hand_evidence.py — snapshots from the track + geometry, and shape costs."""

from __future__ import annotations

import pytest

from migs_tab.fretboard import FrameGeometry, Fretboard
from migs_tab.hand_evidence import (
    OCTAVE_SHIFT_PENALTY,
    Fingertip,
    HandEvidence,
    Snapshot,
    shape_cost,
)


def _snap(*tips: tuple[str, float, int]) -> Snapshot:
    return Snapshot(0.0, [Fingertip(n, fc, s, 0.1) for n, fc, s in tips], 1.0)


class TestShapeCost:
    def test_open_chord_with_fingers_on_the_chord_is_free(self):
        # Open E: index on G fret 1, middle on A fret 2, ring on D fret 2.
        snap = _snap(("index", 0.6, 3), ("middle", 1.5, 1), ("ring", 1.6, 2), ("pinky", 1.9, 4))
        combo = ((0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0))
        assert shape_cost(snap, combo) == 0.0

    def test_notes_up_the_neck_cost_when_fingers_are_open_position(self):
        snap = _snap(("index", 0.6, 3), ("middle", 1.5, 1), ("ring", 1.6, 2), ("pinky", 1.9, 4))
        # The same chord's upper notes pushed to G4 / B5 / e4.
        wrong = ((0, 0), (1, 2), (2, 2), (3, 4), (4, 5), (5, 4))
        assert shape_cost(snap, wrong) > 6.0
        # The octave-corrected reading (G1 instead of e4) is cheap even with the penalty.
        right = ((0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0))
        assert shape_cost(snap, right) + OCTAVE_SHIFT_PENALTY < shape_cost(snap, wrong)

    def test_barre_index_covers_all_strings(self):
        # F#m barre at fret 2: index tip seen over the high e string.
        snap = _snap(("index", 1.5, 5), ("ring", 3.6, 1), ("pinky", 3.5, 2))
        combo = ((0, 2), (1, 4), (2, 4), (3, 2), (4, 2), (5, 2))
        assert shape_cost(snap, combo) == 0.0

    def test_open_strings_never_cost(self):
        snap = _snap(("index", 7.5, 2))
        assert shape_cost(snap, ((0, 0), (1, 0))) == 0.0

    def test_out_of_range_shape_pays_the_hand_range_penalty(self):
        snap = _snap(("index", 7.5, 2), ("middle", 8.5, 3))
        cost = shape_cost(snap, ((2, 2),))
        assert cost == pytest.approx(6.0 + 3.0)  # per-note cap plus the range penalty


def _fb() -> Fretboard:
    g = FrameGeometry(
        t=10.0,
        strings=[[397.0 + 18.0 * i, 0.0] for i in range(6)],
        p0=1200.0,
        p1=-1400.0,
        p2=0.0,
        quality=1.0,
        rms=1.0,
        nut_strength=0.9,
    )
    return Fretboard(1280, 720, True, 2.0, [g])


def _track(t: float, tips_px: list[tuple[float, float]], role: str = "fretting") -> dict:
    lm = [[0.5, 0.5]] * 21
    for i, (x, y) in zip((8, 12, 16, 20), tips_px, strict=True):
        lm[i] = [x / 1280, y / 720]
    return {
        "frame_width": 1280,
        "frame_height": 720,
        "fps": 5.0,
        "frames": [
            {
                "index": 0,
                "t": t,
                "hands": [{"label": "Left", "score": 1.0, "landmarks": lm, "role": role}],
            }
        ],
    }


class TestHandEvidence:
    def test_snapshot_reads_frets_and_strings(self):
        fb = _fb()
        g = fb.frames[0]
        x2 = (g.wire_x(1) + g.wire_x(2)) / 2  # pressing fret 2
        track = _track(10.1, [(x2, 433.0), (x2, 451.0), (x2, 469.0), (x2, 487.0)])
        ev = HandEvidence(track, fb)
        snap = ev.at(10.2)
        assert snap is not None and snap.t == 10.1
        assert snap.index_fret == 2
        assert [f.string for f in snap.fingertips] == [2, 3, 4, 5]

    def test_no_snapshot_far_from_any_tracked_frame(self):
        ev = HandEvidence(_track(10.0, [(900.0, 433.0)] * 4), _fb())
        assert ev.at(11.0) is None

    def test_picking_hand_only_gives_no_snapshot(self):
        ev = HandEvidence(_track(10.0, [(900.0, 433.0)] * 4, role="picking"), _fb())
        assert ev.at(10.0) is None

    def test_load_returns_none_without_files(self, tmp_path):
        from migs_tab.paths import VideoPaths

        assert HandEvidence.load(VideoPaths("aaa11111111", cache_dir=tmp_path)) is None


def test_penalty_is_smaller_than_a_clear_hand_mismatch():
    assert OCTAVE_SHIFT_PENALTY < 2.5
    assert OCTAVE_SHIFT_PENALTY > 0
    snap = _snap(("index", 0.6, 3))
    assert shape_cost(snap, ((5, 4),)) > OCTAVE_SHIFT_PENALTY
    pytest.approx(shape_cost(snap, ((3, 1),)), 0.0)
