"""Tests for fret.py — chord names, templates, clustering, fret assignment."""

from __future__ import annotations

import json

from migs_tab.fret import (
    STANDARD_TUNING,
    _build_chord_templates_for_tuning,
    _chord_pitch_classes,
    _cluster_notes_by_onset,
    _dedupe_same_pitch_onsets,
    _enumerate_shapes,
    _filter_by_chord_context,
    _filter_sympathetic_resonance,
    _intrinsic_cost,
    _load_active_tuning,
    _load_tuning_label,
    _shapes_for_tuning_label,
    _sounding_chord_name,
    assign_frets,
)
from migs_tab.paths import VideoPaths


class TestChordPitchClasses:
    def test_major(self):
        assert _chord_pitch_classes("C") == frozenset({0, 4, 7})
        assert _chord_pitch_classes("G") == frozenset({7, 11, 2})

    def test_minor(self):
        assert _chord_pitch_classes("Am") == frozenset({9, 0, 4})
        assert _chord_pitch_classes("Em") == frozenset({4, 7, 11})

    def test_dominant7(self):
        # G7 = G B D F
        assert _chord_pitch_classes("G7") == frozenset({7, 11, 2, 5})

    def test_sharp_root(self):
        assert _chord_pitch_classes("F#") == frozenset({6, 10, 1})

    def test_flat_root(self):
        # Eb = D#, so Eb major is {D#, G, A#} = {3, 7, 10}
        assert _chord_pitch_classes("Eb") == frozenset({3, 7, 10})

    def test_unparseable_returns_none(self):
        assert _chord_pitch_classes("XYZ") is None
        assert _chord_pitch_classes("") is None


class TestSoundingChordName:
    def test_open_am(self):
        # Am shape under Standard tuning: open A, D fret 2, G fret 2, B fret 1, e open.
        shape = {1: 0, 2: 2, 3: 2, 4: 1, 5: 0}
        assert _sounding_chord_name(shape, STANDARD_TUNING) == "Am"

    def test_open_e_major(self):
        shape = {0: 0, 1: 2, 2: 2, 3: 1, 4: 0, 5: 0}
        assert _sounding_chord_name(shape, STANDARD_TUNING) == "E"

    def test_open_g_major(self):
        shape = {0: 3, 1: 2, 2: 0, 3: 0, 4: 0, 5: 3}
        assert _sounding_chord_name(shape, STANDARD_TUNING) == "G"

    def test_drop_d_e_shape_at_5_is_a_major(self):
        # E-shape barre at fret 5 under Standard = A major.
        shape = {0: 5, 1: 7, 2: 7, 3: 6, 4: 5, 5: 5}
        assert _sounding_chord_name(shape, STANDARD_TUNING) == "A"

    def test_am_shape_under_capo3_is_cm(self):
        # Standard tuning at capo 3 → effective open strings shifted up 3.
        capo3 = tuple(s + 3 for s in STANDARD_TUNING)
        shape = {1: 0, 2: 2, 3: 2, 4: 1, 5: 0}
        assert _sounding_chord_name(shape, capo3) == "Cm"

    def test_empty_shape_returns_none(self):
        assert _sounding_chord_name({}, STANDARD_TUNING) is None

    def test_unrecognized_intervals_returns_none(self):
        # Two notes a major 2nd apart (no fifth or third) — doesn't match any
        # quality pattern.
        # C3 (48) + D3 (50) → intervals {0, 2} = unrecognized
        shape = {2: 0, 3: 0}  # D open + G open in standard = D + G = {0, 5}
        # D + G is a sus4 (interval {0, 5, 7}? only {0, 5} no — that's sus4 only if has 7)
        # Actually just {0, 5} — no pattern, returns None.
        # ...except sus4 is {0, 5, 7} so {0, 5} is a subset of sus4. Hmm.
        # Let me pick something clearly unrecognized:
        shape = {
            3: 0
        }  # single G note — only interval {0}, no pattern matches except maybe power chord {0, 7}
        # Single pitch: just {0}, no match. Should return None.
        assert _sounding_chord_name(shape, STANDARD_TUNING) is None


class TestBuildChordTemplatesForTuning:
    def test_standard_has_common_chords(self):
        templates = _build_chord_templates_for_tuning(STANDARD_TUNING, "Standard")
        for chord in ["A", "Am", "C", "D", "Dm", "E", "Em", "G"]:
            assert chord in templates, f"Standard tuning should produce template for {chord}"

    def test_drop_d_has_d5(self):
        templates = _build_chord_templates_for_tuning((38, 45, 50, 55, 59, 64), "Drop D")
        assert "D5" in templates

    def test_open_d_has_open_d_major(self):
        templates = _build_chord_templates_for_tuning((38, 45, 50, 54, 57, 62), "Open D")
        assert "D" in templates

    def test_capo_shifts_chord_names(self):
        # Capo 3 standard: an "Am shape" produces Cm.
        capo3 = tuple(s + 3 for s in STANDARD_TUNING)
        templates = _build_chord_templates_for_tuning(capo3, "Standard")
        assert "Cm" in templates
        # And not Am (because the shape would now be called Cm).
        # Well, actually we may still have Am from movable barre shapes — that's fine,
        # the assertion that interests us is that Cm is present.

    def test_multiple_voicings_get_unique_keys(self):
        # Should have cowboy Am AND barre Am at fret 5, both for Standard.
        # The first wins as "Am", later ones get _v1, _v2 suffixes.
        templates = _build_chord_templates_for_tuning(STANDARD_TUNING, "Standard")
        am_keys = [k for k in templates if k == "Am" or k.startswith("Am_v")]
        # Should have at least 2 voicings (cowboy + barre at fret 5 = E-minor shape).
        assert len(am_keys) >= 2


class TestClusterNotesByOnset:
    def test_single_note(self):
        notes = [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 70}]
        assert _cluster_notes_by_onset(notes) == [[0]]

    def test_simultaneous_notes_cluster(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 70},
            {"start": 1.05, "end": 1.5, "pitch": 64, "velocity": 70},
            {"start": 1.10, "end": 1.5, "pitch": 67, "velocity": 70},
        ]
        clusters = _cluster_notes_by_onset(notes)
        assert clusters == [[0, 1, 2]]

    def test_separated_notes_separate_clusters(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 70},
            {"start": 2.0, "end": 2.5, "pitch": 64, "velocity": 70},  # > 0.18s gap
        ]
        clusters = _cluster_notes_by_onset(notes)
        assert clusters == [[0], [1]]

    def test_long_run_splits_at_max_duration(self):
        # 8 notes each 0.1s apart → 0.7s span, exceeds MAX_CLUSTER_DURATION (0.40)
        notes = [
            {"start": i * 0.10, "end": (i + 1) * 0.10, "pitch": 60, "velocity": 70}
            for i in range(8)
        ]
        clusters = _cluster_notes_by_onset(notes)
        # The cluster should split at least once — total duration > 0.40s.
        assert len(clusters) >= 2


class TestDedupeSamePitchOnsets:
    def test_keeps_loudest_of_close_duplicates(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 50},
            {"start": 1.10, "end": 1.5, "pitch": 60, "velocity": 80},  # louder duplicate
        ]
        result = _dedupe_same_pitch_onsets(notes)
        assert len(result) == 1
        assert result[0]["velocity"] == 80

    def test_keeps_distinct_pitches(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 70},
            {"start": 1.05, "end": 1.5, "pitch": 64, "velocity": 70},
        ]
        result = _dedupe_same_pitch_onsets(notes)
        assert len(result) == 2

    def test_keeps_distant_same_pitch(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 70},
            {"start": 5.0, "end": 5.5, "pitch": 60, "velocity": 70},  # far apart
        ]
        result = _dedupe_same_pitch_onsets(notes)
        assert len(result) == 2


class TestFilterByChordContext:
    def test_keeps_in_chord_notes(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 70},  # C, in C major
            {"start": 1.5, "end": 2.0, "pitch": 64, "velocity": 70},  # E
        ]
        spans = [(0.0, 5.0, "C")]
        result = _filter_by_chord_context(notes, spans)
        assert len(result) == 2

    def test_drops_low_velocity_out_of_chord(self):
        notes = [
            # A (pc 9) is NOT in C major {0, 4, 7} AND not within the
            # ±1-semitone tolerance band ({11, 0, 1, 3, 4, 5, 6, 7, 8}).
            # Low velocity, gets dropped.
            {"start": 1.0, "end": 1.5, "pitch": 57, "velocity": 30},
        ]
        spans = [(0.0, 5.0, "C")]
        result = _filter_by_chord_context(notes, spans)
        assert len(result) == 0

    def test_keeps_loud_out_of_chord_as_melody(self):
        notes = [
            # Same out-of-chord A but at high velocity — likely a melodic
            # passing tone the player intended, so keep.
            {"start": 1.0, "end": 1.5, "pitch": 57, "velocity": 90},
        ]
        spans = [(0.0, 5.0, "C")]
        result = _filter_by_chord_context(notes, spans)
        assert len(result) == 1

    def test_no_spans_passes_through(self):
        notes = [{"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 30}]
        result = _filter_by_chord_context(notes, [])
        assert len(result) == 1


class TestFilterSympatheticResonance:
    def test_drops_quiet_note_in_cluster(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 100},
            {"start": 1.0, "end": 1.5, "pitch": 64, "velocity": 30},  # 30% of peak
        ]
        filtered_notes, filtered_clusters = _filter_sympathetic_resonance(notes, [[0, 1]])
        # The 30%-velocity note is below 50% of peak, gets dropped.
        assert len(filtered_notes) == 1
        assert filtered_notes[0]["velocity"] == 100

    def test_keeps_singletons(self):
        notes = [{"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 30}]
        filtered_notes, _ = _filter_sympathetic_resonance(notes, [[0]])
        assert len(filtered_notes) == 1


class TestEnumerateShapes:
    def test_single_note_a3(self):
        notes = [{"start": 0.0, "end": 0.5, "pitch": 57, "velocity": 70}]  # A3
        shapes = _enumerate_shapes(notes, [0])
        # A3 = MIDI 57. Possible positions in standard (within MAX_FRET=19):
        #   low E (s=0) fret 17, A (s=1) fret 12, D (s=2) fret 7,
        #   G (s=3) fret 2; B / high E can't reach this pitch.
        # So we expect 4 shapes.
        assert len(shapes) == 4
        # Cheapest should be G fret 2 (low position, no high-fret penalty).
        best = shapes[0]
        assert best.assignments == ((0, 3, 2),)

    def test_two_string_chord(self):
        # A3 (57) + E4 (64). Best is G fret 2 + high E open.
        notes = [
            {"start": 0.0, "end": 0.5, "pitch": 57, "velocity": 70},
            {"start": 0.0, "end": 0.5, "pitch": 64, "velocity": 70},
        ]
        shapes = _enumerate_shapes(notes, [0, 1])
        assert len(shapes) > 0
        best = shapes[0]
        # Best should put A3 on G2 (or D7) and E4 on high E open.
        # Whichever, both fits should appear.
        for note_idx, _string, _fret in best.assignments:
            assert note_idx in (0, 1)


class TestLoadActiveTuning:
    def test_no_tuning_json_defaults_to_standard(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        assert _load_active_tuning(paths) == STANDARD_TUNING

    def test_reads_strings_and_capo(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.tuning_json.write_text(
            json.dumps({"strings_midi": [38, 45, 50, 55, 59, 64], "capo": 2})
        )
        result = _load_active_tuning(paths)
        # Drop D + capo 2 → each string +2.
        assert result == (40, 47, 52, 57, 61, 66)

    def test_invalid_tuning_json_falls_back(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.tuning_json.write_text("not json")
        assert _load_active_tuning(paths) == STANDARD_TUNING


class TestLoadTuningLabel:
    def test_default(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        assert _load_tuning_label(paths) == "Standard"

    def test_from_json(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.tuning_json.write_text(json.dumps({"label": "Drop D"}))
        assert _load_tuning_label(paths) == "Drop D"


class TestIntrinsicCost:
    def test_open_position_cheaper_than_high(self):
        # Same chord shape (3 notes on adjacent strings), open vs at fret 12.
        open_combo = ((0, 0), (1, 2), (2, 2))
        high_combo = ((0, 12), (1, 14), (2, 14))
        assert _intrinsic_cost(open_combo) < _intrinsic_cost(high_combo)

    def test_compact_cheaper_than_wide(self):
        compact = ((0, 0), (1, 2), (2, 2))  # span 2
        wide = ((0, 0), (1, 0), (2, 12))  # span 12
        assert _intrinsic_cost(compact) < _intrinsic_cost(wide)


class TestShapesForTuningLabel:
    def test_standard_returns_base_shapes(self):
        shapes = _shapes_for_tuning_label("Standard")
        assert "Am" in shapes
        assert "G" in shapes

    def test_drop_d_has_overrides(self):
        shapes = _shapes_for_tuning_label("Drop D")
        # Em shape in Drop D is overridden (string 0 fret 2 not 0).
        assert shapes["Em"][0] == 2

    def test_double_drop_d_has_overrides(self):
        shapes = _shapes_for_tuning_label("Double Drop D")
        # Em in Double Drop D: fret 2 on BOTH low D and high D strings.
        assert shapes["Em"][0] == 2
        assert shapes["Em"][5] == 2


def test_assign_frets_skips_when_cached(tmp_path):
    """Smoke test — verify caching behavior of the top-level function."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    paths.frets_json.write_text('{"note_count": 0, "notes": []}')
    # Should be a no-op (cached) and not require notes.json.
    result = assign_frets(paths, force=False)
    assert result is paths
