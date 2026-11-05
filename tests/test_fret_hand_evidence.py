"""Tests for the hand-evidence terms in fret.py's shape enumeration."""

from __future__ import annotations

from migs_tab import fret
from migs_tab.hand_evidence import Fingertip, Snapshot


def _snap(*tips: tuple[str, float, int]) -> Snapshot:
    return Snapshot(0.0, [Fingertip(n, fc, s, 0.1) for n, fc, s in tips], 1.0)


def _shapes(pitches: list[int], snap: Snapshot | None):
    notes = [{"pitch": p, "start": 1.0, "end": 1.5} for p in pitches]
    return fret._enumerate_shapes(notes, list(range(len(notes))), snap)


def _placements(shape) -> list[tuple[int, int]]:
    return [(s, f) for _, s, f in shape.assignments]


class TestHandEvidenceInEnumeration:
    def setup_method(self):
        fret._ACTIVE_TUNING = fret.STANDARD_TUNING
        fret._CHORD_TEMPLATES = fret._build_chord_templates_for_tuning(
            fret.STANDARD_TUNING, "Standard"
        )

    def test_open_e_chord_fingers_pull_a_wrong_octave_down(self):
        # Open E with index on G1, middle on A2, ring on D2; the transcriber
        # heard G#4 (68) instead of G#3 (56).
        snap = _snap(("index", 0.6, 3), ("middle", 1.5, 1), ("ring", 1.6, 2), ("pinky", 1.9, 4))
        best = _shapes([40, 47, 52, 68, 59, 64], snap)[0]
        assert best.octave_shifts == ((3, -12),)
        assert (3, 1) in _placements(best)  # G#3 on the G string, fret 1
        assert (5, 4) not in _placements(best)

    def test_supported_raw_reading_is_never_re_read(self):
        # F#m barre at fret 2 with the ring finger on D4: a real F#3 (54) has a
        # supported placement, so no octave option is generated even though the
        # index barre would "support" F#4 on the high e.
        snap = _snap(("index", 1.5, 5), ("ring", 3.6, 2), ("pinky", 3.5, 1))
        shapes = _shapes([45, 54, 57], snap)
        assert all(sh.octave_shifts == () for sh in shapes)
        assert (2, 4) in _placements(shapes[0])

    def test_without_snapshot_nothing_changes(self):
        shapes = _shapes([40, 47, 52, 68, 59, 64], None)
        assert all(sh.octave_shifts == () for sh in shapes)

    def test_pitch_in_records_follows_the_placement(self):
        # The effective pitch of a shifted note is tuning[string] + fret.
        snap = _snap(("index", 0.6, 3), ("middle", 1.5, 1), ("ring", 1.6, 2), ("pinky", 1.9, 4))
        best = _shapes([68], snap)[0]
        (_, s, f), *_ = best.assignments
        assert fret.STANDARD_TUNING[s] + f == 56
