"""Tests for the hand-evidence terms in fret.py's shape enumeration."""

from __future__ import annotations

from collections import Counter

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


class TestUnfingeredOvertones:
    def setup_method(self):
        fret._ACTIVE_TUNING = fret.STANDARD_TUNING
        fret._CHORD_TEMPLATES = fret._build_chord_templates_for_tuning(
            fret.STANDARD_TUNING, "Standard"
        )

    # Shuffle on E: index on A fret 2, the other fingers hovering near frets 3-5.
    SHUFFLE = (("index", 2.0, 1), ("middle", 3.0, 1), ("ring", 3.7, 1), ("pinky", 4.6, 3))
    # E2 and B2 as played, plus E3, B3, E4 and G#4: partials 2-5 of the low E string.
    PARTIALS_OF_E2 = [40, 47, 52, 59, 64, 68]

    def test_overtone_with_no_finger_near_is_left_out(self):
        best = _shapes(self.PARTIALS_OF_E2, _snap(*self.SHUFFLE))[0]
        assert best.dropped == (5,)  # G#4: no open string left and no finger near fret 4 on e
        assert _placements(best) == [(0, 0), (1, 2), (2, 2), (4, 0), (5, 0)]

    def test_fingered_octave_above_the_bass_is_kept(self):
        # Am: A2 open, A3 under the ring finger on the G string, fret 2.
        snap = _snap(("index", 0.5, 4), ("middle", 1.5, 2), ("ring", 1.6, 3), ("pinky", 2.4, 4))
        best = _shapes([45, 52, 57, 60, 64], snap)[0]
        assert best.dropped == ()
        assert (3, 2) in _placements(best)

    def test_nothing_is_left_out_without_hand_evidence(self):
        shapes = _shapes(self.PARTIALS_OF_E2, None)
        assert all(sh.dropped == () for sh in shapes)

    def test_note_unrelated_to_a_lower_note_is_never_left_out(self):
        # C#4 over E2 is no partial of E2: the decode must place it even with no finger near.
        shapes = _shapes([40, 61], _snap(*self.SHUFFLE))
        assert all(sh.dropped == () for sh in shapes)


class TestSeenElsewhere:
    """Clusters without a snapshot use placements the hand confirmed elsewhere."""

    def setup_method(self):
        fret._ACTIVE_TUNING = fret.STANDARD_TUNING
        fret._CHORD_TEMPLATES = fret._build_chord_templates_for_tuning(
            fret.STANDARD_TUNING, "Standard"
        )

    @staticmethod
    def _shapes(pitches: list[int], seen: dict | None):
        notes = [{"pitch": p, "start": 1.0, "end": 1.5} for p in pitches]
        return fret._enumerate_shapes(notes, list(range(len(notes))), None, seen)

    def test_pitch_goes_where_the_hand_was_seen_playing_it(self):
        assert _placements(self._shapes([64], None)[0]) == [(5, 0)]  # open e by default
        seen = {64: Counter({(4, 5): 9, (5, 0): 1})}
        assert _placements(self._shapes([64], seen)[0]) == [(4, 5)]

    def test_octave_reading_the_hand_confirmed_is_offered(self):
        seen = {68: Counter({(3, 1): 8, (5, 4): 2})}  # G#4 was G#3 on the G string 8 times of 10
        best = self._shapes([68], seen)[0]
        assert _placements(best) == [(3, 1)]
        assert best.octave_shifts == ((0, -12),)

    def test_too_few_sightings_change_nothing(self):
        seen = {64: Counter({(4, 5): 3})}
        assert _placements(self._shapes([64], seen)[0]) == [(5, 0)]

    def test_sightings_count_only_placements_a_snapshot_confirms(self):
        snap = _snap(("index", 0.5, 4), ("middle", 1.5, 2), ("ring", 1.6, 3), ("pinky", 2.4, 4))
        notes = [{"pitch": p, "start": 1.0, "end": 1.5} for p in (45, 57, 69)]
        on_finger = fret.Shape(
            ((0, 1, 0), (1, 3, 2), (2, 5, 5)), 0.0
        )  # A0, G2, and e5 with no finger
        seen = fret._seen_placements(
            notes, [0, 1], [([0, 1, 2], []), ([0], [])], [on_finger, on_finger], [snap, None]
        )
        assert seen == {45: Counter({(1, 0): 1}), 57: Counter({(3, 2): 1})}
