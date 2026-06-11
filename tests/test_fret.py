"""Tests for fret.py — chord names, templates, clustering, fret assignment."""

from __future__ import annotations

import json

import numpy as np

from migs_tab.fret import (
    STANDARD_TUNING,
    _build_chord_templates_for_tuning,
    _chord_pitch_classes,
    _cluster_notes_by_onset,
    _dedupe_same_pitch_onsets,
    _enumerate_shapes,
    _filter_by_chord_context,
    _filter_harmonic_overtones,
    _filter_sympathetic_resonance,
    _intrinsic_cost,
    _load_active_tuning,
    _load_tuning_label,
    _min_playable_fret,
    _octave_alternative,
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

    def test_repeated_pitch_starts_new_cluster(self):
        # Regression (Angie bridge_main_lick t=1233.4-1234.4 s): a 67/69
        # trill at 0.07 s gaps chains into one cluster via the rolling gap;
        # the repeated pitches made the cluster unfingerable (one distinct
        # string per note) and the WHOLE cluster — chord stabs included —
        # was silently dropped as unplayable. A same-pitch repeat is a
        # sequential re-articulation (dedupe already merged split
        # detections), so it must start a NEW cluster instead.
        notes = [
            {"start": 0.00, "end": 0.07, "pitch": 67, "velocity": 70},
            {"start": 0.07, "end": 0.14, "pitch": 69, "velocity": 70},
            {"start": 0.14, "end": 0.21, "pitch": 67, "velocity": 70},  # repeat → split
            {"start": 0.21, "end": 0.28, "pitch": 69, "velocity": 70},
        ]
        clusters = _cluster_notes_by_onset(notes)
        assert clusters == [[0, 1], [2, 3]]
        for cluster in clusters:
            pitches = [notes[i]["pitch"] for i in cluster]
            assert len(pitches) == len(set(pitches))

    def test_split_quiet_rearticulation_bypasses_sympathetic_gate(self):
        # Disclosure pin (review nit), not an endorsement: the
        # duplicate-pitch split severs a trailing quiet same-pitch repeat
        # from the strum, so the sympathetic gate compares the new cluster
        # only against its OWN all-quiet peak — pre-split, the single
        # cluster's floor (95 * 0.5 = 47.5) dropped the quiet pair. On
        # pseudo-velocity runs render's quiet-note floor still catches
        # them; on plain basic-pitch runs they now print. Pinned so the
        # trade-off stays visible if the gate or split changes.
        notes = [
            {"start": 0.00, "end": 0.50, "pitch": 55, "velocity": 95},
            {"start": 0.02, "end": 0.50, "pitch": 67, "velocity": 71},
            {"start": 0.04, "end": 0.50, "pitch": 62, "velocity": 80},
            # 67 repeat: 0.09 s after its first strike (outside the dedupe
            # window) but only 0.07 s after the last onset (inside the
            # rolling gap) — the pitch rule, not the gap, splits here.
            {"start": 0.11, "end": 0.20, "pitch": 67, "velocity": 8},
            {"start": 0.17, "end": 0.25, "pitch": 64, "velocity": 6},
        ]
        clusters = _cluster_notes_by_onset(notes)
        assert clusters == [[0, 1, 2], [3, 4]]
        filtered, _ = _filter_sympathetic_resonance(notes, clusters)
        assert len(filtered) == 5  # quiet pair survives against its own peak of 8


class TestDedupeSamePitchOnsets:
    def test_keeps_loudest_of_close_duplicates(self):
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 50},
            {"start": 1.05, "end": 1.5, "pitch": 60, "velocity": 80},  # louder duplicate
        ]
        result = _dedupe_same_pitch_onsets(notes)
        assert len(result) == 1
        assert result[0]["velocity"] == 80

    def test_keeps_restrums_just_outside_cluster_gap(self):
        # Regression: the window was 0.20s, which deleted genuine 16th-note
        # re-strums (16ths at 120 bpm are 0.125s apart). Two same-pitch notes
        # 0.12s apart are separate articulations and must BOTH survive.
        notes = [
            {"start": 1.0, "end": 1.1, "pitch": 60, "velocity": 70},
            {"start": 1.12, "end": 1.22, "pitch": 60, "velocity": 70},
        ]
        result = _dedupe_same_pitch_onsets(notes)
        assert len(result) == 2

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

    def test_bass_root_exempt_when_pseudo_velocities(self):
        # Regression (Angie t=903.2 s open-E7 demo): the REAL low-E pedal
        # (pitch 40) measured pseudo-velocity 33 vs cluster peak 84 — the
        # CQT under-measures low strings — and the gate deleted it along
        # with the genuine phantoms. On pseudo-velocity (MT3 + stem) runs
        # the caller sets bass_exempt=True and the LOWEST pitch of each
        # cluster must survive; the gate must still drop quiet NON-bass
        # members.
        notes = [
            {"start": 903.2, "end": 903.5, "pitch": 40, "velocity": 33},  # bass: exempt
            {"start": 903.2, "end": 903.5, "pitch": 52, "velocity": 80},
            {"start": 903.2, "end": 903.5, "pitch": 59, "velocity": 84},  # peak
            {"start": 903.2, "end": 903.5, "pitch": 64, "velocity": 11},  # phantom: drop
        ]
        filtered, _ = _filter_sympathetic_resonance(notes, [[0, 1, 2, 3]], bass_exempt=True)
        assert sorted(n["pitch"] for n in filtered) == [40, 52, 59]

    def test_quiet_bass_dropped_without_pseudo_velocities(self):
        # The exemption compensates for a CQT low-string bias that
        # basic-pitch confidence velocities do not have, so by default the
        # gate must still drop a quiet lowest member: a strong strum makes
        # the other open strings ring quietly (this filter's core target),
        # and at velocity 40 this open low E would otherwise clear
        # render's quiet-note floor (35) and print.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 40, "velocity": 40},  # ring: drop
            {"start": 1.0, "end": 1.5, "pitch": 76, "velocity": 100},
            {"start": 1.0, "end": 1.5, "pitch": 79, "velocity": 90},
        ]
        filtered, _ = _filter_sympathetic_resonance(notes, [[0, 1, 2]])
        assert sorted(n["pitch"] for n in filtered) == [76, 79]

    def test_sub_octave_ghost_dropped_without_pseudo_velocities(self):
        # A basic-pitch sub-octave ghost (p-12 of the real bass) is by
        # construction the cluster's LOWEST pitch; the overtone gates only
        # look upward (+12/+19/+24) and the chord-context filter can't
        # catch it (same pitch class as the real bass) — this gate is the
        # only defense, so the bass exemption must not shield it on
        # native-velocity runs.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 40, "velocity": 20},  # ghost of 52
            {"start": 1.0, "end": 1.5, "pitch": 52, "velocity": 90},
            {"start": 1.0, "end": 1.5, "pitch": 59, "velocity": 80},
        ]
        filtered, _ = _filter_sympathetic_resonance(notes, [[0, 1, 2]])
        assert sorted(n["pitch"] for n in filtered) == [52, 59]


class TestFilterHarmonicOvertones:
    def test_drops_octave_harmonic(self):
        # A strong A3 (pitch 57) plus a quiet A4 (pitch 69, +12) in the same
        # cluster — the A4 looks like the 2nd harmonic and gets dropped.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 57, "velocity": 100},
            {"start": 1.05, "end": 1.5, "pitch": 69, "velocity": 50},
        ]
        filtered, _ = _filter_harmonic_overtones(notes, [[0, 1]])
        assert len(filtered) == 1
        assert filtered[0]["pitch"] == 57

    def test_drops_octave_plus_fifth(self):
        # +19 semitones is the 3rd harmonic.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 45, "velocity": 100},  # A2
            {"start": 1.0, "end": 1.5, "pitch": 64, "velocity": 40},  # A2 + 19 = E4
        ]
        filtered, _ = _filter_harmonic_overtones(notes, [[0, 1]])
        assert len(filtered) == 1
        assert filtered[0]["pitch"] == 45

    def test_keeps_loud_octave(self):
        # If the +12 note is LOUDER (ratio >= 0.7), keep it as a real note.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 57, "velocity": 80},
            {"start": 1.05, "end": 1.5, "pitch": 69, "velocity": 80},  # equal velocity
        ]
        filtered, _ = _filter_harmonic_overtones(notes, [[0, 1]])
        assert len(filtered) == 2

    def test_keeps_non_harmonic_intervals(self):
        # +7 (perfect 5th) is not a typical overtone — keep both.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 100},
            {"start": 1.0, "end": 1.5, "pitch": 67, "velocity": 30},
        ]
        filtered, _ = _filter_harmonic_overtones(notes, [[0, 1]])
        assert len(filtered) == 2

    def test_skip_octave_keeps_genuine_octave_pair(self):
        # Regression (Angie t=42.16 s, chroma-verified intro): the genuine
        # F 53/65 octave pair carried pseudo-velocities 104/68 → ratio
        # 0.654 < _HARMONIC_VELOCITY_RATIO, so this gate deleted the upper
        # note AFTER the calibrated octave pass had decided to keep it.
        # With skip_octave=True (audio-evidence pass ran) +12 pairs are
        # off-limits here and both notes must survive.
        notes = [
            {"start": 42.16, "end": 42.9, "pitch": 53, "velocity": 104},
            {"start": 42.26, "end": 42.9, "pitch": 65, "velocity": 68},
        ]
        filtered, _ = _filter_harmonic_overtones(notes, [[0, 1]], skip_octave=True)
        assert sorted(n["pitch"] for n in filtered) == [53, 65]

    def test_skip_octave_still_drops_higher_harmonics(self):
        # +19 (3rd harmonic) is NOT covered by salience's octave test, so it
        # stays adjudicated here even when the audio-evidence pass ran.
        notes = [
            {"start": 1.0, "end": 1.5, "pitch": 45, "velocity": 100},
            {"start": 1.0, "end": 1.5, "pitch": 64, "velocity": 40},
        ]
        filtered, _ = _filter_harmonic_overtones(notes, [[0, 1]], skip_octave=True)
        assert [n["pitch"] for n in filtered] == [45]

    def test_skips_singletons(self):
        notes = [{"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 30}]
        filtered, _ = _filter_harmonic_overtones(notes, [[0]])
        assert len(filtered) == 1


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


class TestMinPlayableFret:
    def test_open_string(self):
        assert _min_playable_fret(64) == 0  # high E open

    def test_high_only_pitch(self):
        # A5 = 81: only reachable on the high E string at fret 17.
        assert _min_playable_fret(81) == 17

    def test_out_of_range(self):
        assert _min_playable_fret(20) is None  # below low E
        assert _min_playable_fret(100) is None  # above fret 19 everywhere


class TestOctaveAlternative:
    def test_single_note_offers_lower_octave(self):
        # A3 (57) chosen at G-string fret 2; the octave-down reading (45)
        # is the open A string — must be offered with an octave_shift label.
        notes = [{"start": 0.0, "end": 0.5, "pitch": 57, "velocity": 70}]
        chosen = _enumerate_shapes(notes, [0])[0]
        alt = _octave_alternative(notes, [0], chosen)
        assert alt is not None
        # local_note_index is CLUSTER-LOCAL (overrides-schema namespace),
        # not the global notes[]-list index.
        assert alt["octave_shift"]["local_note_index"] == 0
        assert alt["octave_shift"]["pitch_delta"] == -12
        assert alt["octave_shift"]["new_pitch"] == 45
        assert alt["assignments"] == [{"string": 1, "fret": 0}]

    def test_skips_shift_onto_existing_cluster_pitch(self):
        # Cluster already contains both 57 and 69 — shifting either onto the
        # other would claim a unison double-stop; the only legal shifts are
        # 57→45 / 69→81, so whatever is returned must not target 57 or 69.
        notes = [
            {"start": 0.0, "end": 0.5, "pitch": 57, "velocity": 70},
            {"start": 0.0, "end": 0.5, "pitch": 69, "velocity": 70},
        ]
        chosen = _enumerate_shapes(notes, [0, 1])[0]
        alt = _octave_alternative(notes, [0, 1], chosen)
        assert alt is not None
        assert alt["octave_shift"]["new_pitch"] not in (57, 69)


def _write_notes(paths: VideoPaths, notes: list[dict]) -> None:
    paths.notes_mt3_json.write_text(json.dumps({"notes": notes}))


def test_cluster_ids_share_one_namespace_with_unplayable(tmp_path):
    """Regression: unplayable_clusters[] used original cluster ids while kept
    notes were renumbered — overrides and the vision pass could mis-target.
    Now an unplayable cluster's id must never be reused by a kept cluster."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    _write_notes(
        paths,
        [
            {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100},
            # Pitch 24 is below low E on every string — unplayable cluster.
            {"start": 1.0, "end": 1.5, "pitch": 24, "velocity": 100},
            {"start": 2.0, "end": 2.5, "pitch": 64, "velocity": 100},
        ],
    )
    assign_frets(paths, force=True)
    out = json.loads(paths.frets_json.read_text())
    assert len(out["unplayable_clusters"]) == 1
    unplayable_id = out["unplayable_clusters"][0]["cluster_id"]
    assert unplayable_id == 1
    kept_note_ids = {n["cluster_id"] for n in out["notes"]}
    kept_record_ids = {c["cluster_id"] for c in out["clusters"]}
    assert unplayable_id not in kept_note_ids
    assert kept_note_ids == kept_record_ids == {0, 2}


def test_no_stem_means_no_velocity_and_no_artifacts(tmp_path):
    """Without a guitar stem the audio-evidence pass must be skipped: output
    format identical to the pre-salience one."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    _write_notes(paths, [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100}])
    assign_frets(paths, force=True)
    out = json.loads(paths.frets_json.read_text())
    assert "overtone_artifacts" not in out
    assert out["params"]["audio_evidence"] is False
    assert all("velocity" not in n for n in out["notes"])


def test_trill_chained_into_chord_cluster_survives(tmp_path):
    """End-to-end regression for the Angie bridge over-deletion: a G-chord
    stab whose onset chain runs straight into a 67/69 trill previously formed
    one duplicate-pitch cluster with NO playable shape — assign_frets dropped
    all of it (chord included) as unplayable. With the duplicate-pitch split
    every note must survive and nothing may be reported unplayable."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    _write_notes(
        paths,
        [
            {"start": 0.00, "end": 1.0, "pitch": 55, "velocity": 95},
            {"start": 0.01, "end": 1.0, "pitch": 62, "velocity": 80},
            {"start": 0.02, "end": 1.0, "pitch": 67, "velocity": 71},
            {"start": 0.06, "end": 0.12, "pitch": 69, "velocity": 77},
            # Same-pitch repeats 0.11/0.14 s after their first strikes —
            # outside _SAME_PITCH_DEDUPE_WINDOW (genuine re-articulations)
            # but chained into the same onset run by the 0.07 s gaps.
            {"start": 0.13, "end": 0.19, "pitch": 67, "velocity": 76},  # repeat
            {"start": 0.20, "end": 0.26, "pitch": 69, "velocity": 73},  # repeat
        ],
    )
    assign_frets(paths, force=True)
    out = json.loads(paths.frets_json.read_text())
    assert out["unplayable_clusters"] == []
    assert sorted(n["pitch"] for n in out["notes"]) == [55, 62, 67, 67, 69, 69]


class TestSalvageUnplayableCluster:
    """A cluster with no playable shape must shed only the blocking members,
    not delete its real notes wholesale (Angie lost 394 notes this way)."""

    def test_out_of_range_member_no_longer_nukes_cluster(self, tmp_path):
        # Angie t=397.56 s: an out-of-range D2 (38, below low E) survived the
        # sympathetic gate via the bass exemption and made the whole cluster
        # unplayable — deleting the real {50, 62, 78}. Only the 38 may go.
        # (The exemption is now restricted to stem-backed runs, so this
        # no-stem test gives the 38 a velocity above the sympathetic floor —
        # the salvage path must still be the one that removes it.)
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        _write_notes(
            paths,
            [
                {"start": 0.00, "end": 0.5, "pitch": 38, "velocity": 50},
                {"start": 0.00, "end": 0.5, "pitch": 50, "velocity": 77},
                {"start": 0.01, "end": 0.5, "pitch": 62, "velocity": 68},
                {"start": 0.01, "end": 0.5, "pitch": 78, "velocity": 50},
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert sorted(n["pitch"] for n in out["notes"]) == [50, 62, 78]
        assert out["unplayable_clusters"] == []
        (dropped,) = out["unplayable_notes"]
        assert dropped["pitch"] == 38
        assert "out of range" in dropped["reason"]

    def test_unfingerable_quiet_member_dropped_not_cluster(self, tmp_path):
        # Angie t=887.8 s: quiet bass B2 (47, bass-exempt from the
        # sympathetic gate on that stem-backed run) cannot be fingered
        # within MAX_HAND_SPAN of the real B3/B5 (59@70 open, 83@67 fret
        # 19) — pre-salvage all three vanished. The weakest member goes;
        # the pair stays. (Velocity raised from the measured 14 to 40 —
        # above the 35 sympathetic floor — because this no-stem test has
        # no bass exemption; salvage must do the dropping.)
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        _write_notes(
            paths,
            [
                {"start": 0.00, "end": 0.3, "pitch": 47, "velocity": 40},
                {"start": 0.00, "end": 0.3, "pitch": 59, "velocity": 70},
                {"start": 0.01, "end": 0.3, "pitch": 83, "velocity": 67},
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert sorted(n["pitch"] for n in out["notes"]) == [59, 83]
        (dropped,) = out["unplayable_notes"]
        assert dropped["pitch"] == 47
        assert "No playable shape" in dropped["reason"]

    def test_seven_note_cluster_keeps_six_strongest(self, tmp_path):
        # More distinct pitches than strings: keep a playable 6 (equal
        # velocities tie-break toward dropping the highest pitch).
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        _write_notes(
            paths,
            [
                {"start": i * 0.01, "end": 0.5, "pitch": p, "velocity": 100}
                for i, p in enumerate([40, 45, 50, 55, 59, 64, 65])
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert sorted(n["pitch"] for n in out["notes"]) == [40, 45, 50, 55, 59, 64]
        (dropped,) = out["unplayable_notes"]
        assert dropped["pitch"] == 65

    def test_singleton_out_of_range_still_whole_cluster_unplayable(self, tmp_path):
        # Nothing to salvage in a 1-note cluster — old path preserved.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        _write_notes(paths, [{"start": 0.0, "end": 0.5, "pitch": 24, "velocity": 100}])
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert out["notes"] == []
        assert out["unplayable_notes"] == []
        assert len(out["unplayable_clusters"]) == 1

    def test_all_members_out_of_range_falls_back_to_whole_cluster(self, tmp_path):
        # Salvage returns kept=[] when EVERY member is out of range; the
        # `if kept:` guard must fall through to the old whole-cluster
        # unplayable_clusters record and emit NO unplayable_notes entries
        # (the per-note records exist only when cluster-mates were saved).
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        _write_notes(
            paths,
            [
                {"start": 0.00, "end": 0.5, "pitch": 20, "velocity": 100},
                {"start": 0.01, "end": 0.5, "pitch": 24, "velocity": 100},
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert out["notes"] == []
        assert out["unplayable_notes"] == []
        (cluster,) = out["unplayable_clusters"]
        assert sorted(cluster["pitches"]) == [20, 24]


def test_params_record_backend_provenance(tmp_path):
    """frets.json params must name the backend whose notes were actually
    used plus the source file, so verify's agreement tiers can label
    correctly instead of assuming mt3."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    _write_notes(paths, [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100}])
    assign_frets(paths, force=True)
    out = json.loads(paths.frets_json.read_text())
    assert out["params"]["backend"] == "mt3"
    assert out["params"]["source_notes_file"] == paths.notes_mt3_json.name


def test_params_backend_reflects_fallback(tmp_path):
    """Requesting mt3 with only notes.json on disk falls back to
    basic_pitch — provenance must record what was USED, not what was
    asked for."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    paths.notes_json.write_text(
        json.dumps({"notes": [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100}]})
    )
    assign_frets(paths, force=True, backend="mt3")
    out = json.loads(paths.frets_json.read_text())
    assert out["params"]["backend"] == "basic_pitch"
    assert out["params"]["source_notes_file"] == paths.notes_json.name


def test_params_backend_reflects_fallback_to_mt3(tmp_path):
    """The reverse fallback direction: requesting basic_pitch with only
    notes.mt3.json on disk must record mt3 as the backend actually used."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    _write_notes(paths, [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100}])
    assign_frets(paths, force=True, backend="basic_pitch")
    out = json.loads(paths.frets_json.read_text())
    assert out["params"]["backend"] == "mt3"
    assert out["params"]["source_notes_file"] == paths.notes_mt3_json.name


def test_basic_pitch_no_stem_quiet_bass_still_gated(tmp_path):
    """End-to-end guard for the bass-exemption scope (major review fix):
    a no-stem basic-pitch run writes NO velocities into frets.json, so
    render has no quiet-note backstop — the sympathetic gate itself must
    keep dropping the quiet lowest cluster member there. The exemption is
    reserved for CQT pseudo-velocity (MT3 + stem) runs."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    paths.notes_json.write_text(
        json.dumps(
            {
                "notes": [
                    {"start": 1.00, "end": 1.5, "pitch": 40, "velocity": 40},
                    {"start": 1.01, "end": 1.5, "pitch": 76, "velocity": 100},
                    {"start": 1.02, "end": 1.5, "pitch": 79, "velocity": 90},
                ]
            }
        )
    )
    assign_frets(paths, force=True, backend="basic_pitch")
    out = json.loads(paths.frets_json.read_text())
    assert sorted(n["pitch"] for n in out["notes"]) == [76, 79]
    assert all("velocity" not in n for n in out["notes"])


def test_ambiguous_cluster_lists_octave_alternative(tmp_path):
    """A near-tie single note (G#4 = 68: B-string fret 9 vs high-E fret 4,
    cost delta 0.05) is flagged ambiguous and must offer the octave reading
    (G#3 = 56) as an extra alternative."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    _write_notes(paths, [{"start": 0.0, "end": 0.5, "pitch": 68, "velocity": 100}])
    assign_frets(paths, force=True)
    out = json.loads(paths.frets_json.read_text())
    (cluster,) = out["clusters"]
    assert cluster["ambiguous"] is True
    octave_alts = [a for a in cluster["alternatives"] if "octave_shift" in a]
    assert len(octave_alts) == 1
    assert octave_alts[0]["octave_shift"]["new_pitch"] == 56


class TestAudioEvidencePass:
    """End-to-end assign_frets against a synthetic guitar stem."""

    SR = 22050

    def _write_stem(self, paths: VideoPaths, upper_amp: float) -> None:
        """Two seconds of A4 (440 Hz) plus an A5 (880 Hz) partial at
        ``upper_amp`` relative amplitude — the E(81)/E(69) CQT ratio tracks
        the amplitude ratio (measured: amp 0.05 → ratio 0.035, amp 0.35 →
        ratio 0.247)."""
        import soundfile as sf

        t = np.arange(int(self.SR * 2.0)) / self.SR
        y = 0.5 * (np.sin(2 * np.pi * 440.0 * t) + upper_amp * np.sin(2 * np.pi * 880.0 * t))
        paths.stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(paths.guitar_stem, y.astype(np.float32), self.SR)

    def _octave_pair_notes(self, lower: int, upper: int) -> list[dict]:
        return [
            {"start": 0.5, "end": 1.5, "pitch": lower, "velocity": 100},
            {"start": 0.52, "end": 1.5, "pitch": upper, "velocity": 100},
        ]

    def test_velocity_written_for_every_note(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_stem(paths, upper_amp=0.8)
        _write_notes(paths, self._octave_pair_notes(69, 81))
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert out["params"]["audio_evidence"] is True
        assert len(out["notes"]) > 0
        for n in out["notes"]:
            assert isinstance(n["velocity"], int)
            assert 0 <= n["velocity"] <= 127

    def test_basic_pitch_source_keeps_native_velocities(self, tmp_path):
        # basic-pitch velocities are model confidences — the quantity the
        # velocity gates were originally tuned on — so a stem-backed run
        # must NOT replace them with CQT energy proxies (the overtone pass
        # still runs off the contexts). The pseudo-velocity for a full-
        # amplitude 440 Hz claim would be near 127, so velocity 77
        # surviving proves preservation.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_stem(paths, upper_amp=0.8)
        paths.notes_json.write_text(
            json.dumps({"notes": [{"start": 0.5, "end": 1.5, "pitch": 69, "velocity": 77}]})
        )
        assign_frets(paths, force=True, backend="basic_pitch")
        out = json.loads(paths.frets_json.read_text())
        assert out["params"]["audio_evidence"] is True
        assert [n["velocity"] for n in out["notes"]] == [77]

    def test_strict_overtone_dropped_and_recorded(self, tmp_path):
        # E(81)/E(69) ≈ 0.035 < OCTAVE_ARTIFACT_RATIO → 81 is a phantom.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_stem(paths, upper_amp=0.05)
        _write_notes(paths, self._octave_pair_notes(69, 81))
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert [n["pitch"] for n in out["notes"]] == [69]
        assert len(out["overtone_artifacts"]) == 1
        artifact = out["overtone_artifacts"][0]
        assert artifact["pitch"] == 81
        assert artifact["start"] == 0.52

    def test_gray_zone_overtone_dropped_when_only_playable_high(self, tmp_path):
        # E(81)/E(69) ≈ 0.25 — inside the gray zone — and A5 is only
        # playable at fret 17, so it must still be dropped (the Angie
        # t=828.4s flagship case).
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_stem(paths, upper_amp=0.35)
        _write_notes(paths, self._octave_pair_notes(69, 81))
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert [n["pitch"] for n in out["notes"]] == [69]
        assert len(out["overtone_artifacts"]) == 1
        assert "fret" in out["overtone_artifacts"][0]["reason"]

    def test_gray_zone_octave_kept_when_playable_low(self, tmp_path):
        # Same gray-zone energy ratio as the dropped case above, but claimed
        # as the pair (A3=57, A4=69) over a 220+440 Hz stem: the upper note
        # is playable at fret 5 (G string), i.e. at or below
        # _OCTAVE_GRAY_ZONE_MIN_FRET, so it's a genuine octave pair — kept.
        import soundfile as sf

        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        t = np.arange(int(self.SR * 2.0)) / self.SR
        y = 0.5 * (np.sin(2 * np.pi * 220.0 * t) + 0.45 * np.sin(2 * np.pi * 440.0 * t))
        paths.stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(paths.guitar_stem, y.astype(np.float32), self.SR)
        _write_notes(paths, self._octave_pair_notes(57, 69))
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert sorted(n["pitch"] for n in out["notes"]) == [57, 69]
        assert out["overtone_artifacts"] == []

    def test_pseudo_velocities_revive_sympathetic_gate(self, tmp_path):
        # A3 (220 Hz) at full amplitude plus an E4=64 partial 40 dB down:
        # pseudo-velocities land around 127 vs 42 (ratio ~0.33, below
        # _SYMPATHETIC_RATIO), so the quiet note must be dropped by the
        # revived sympathetic gate. +7 semitones is no harmonic offset and
        # 64 has no p-12 partner in the cluster, so neither the harmonic
        # gate nor the octave pass can be the one dropping it — and
        # overtone_artifacts stays empty to prove that.
        import soundfile as sf

        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        t = np.arange(int(self.SR * 2.0)) / self.SR
        y = 0.5 * (np.sin(2 * np.pi * 220.0 * t) + 0.01 * np.sin(2 * np.pi * 329.63 * t))
        paths.stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(paths.guitar_stem, y.astype(np.float32), self.SR)
        _write_notes(
            paths,
            [
                {"start": 0.5, "end": 1.5, "pitch": 57, "velocity": 100},
                {"start": 0.52, "end": 1.5, "pitch": 64, "velocity": 100},
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert [n["pitch"] for n in out["notes"]] == [57]
        assert out["overtone_artifacts"] == []

    def test_velocity_gate_does_not_undo_calibrated_octave_keep(self, tmp_path):
        # Regression for the harmonic-overtone gate double-adjudicating +12
        # pairs: a louder unrelated burst earlier in the window drags BOTH
        # pair velocities down the dB scale, shrinking their velocity RATIO
        # (measured: 36/59 = 0.61 < _HARMONIC_VELOCITY_RATIO) while the
        # pair's CQT energy ratio stays in the calibrated gray zone
        # (measured 0.283) with the upper note playable at fret 1 → the
        # calibrated pass keeps it. Pre-fix, the velocity gate then deleted
        # the upper note anyway; with the audio-evidence pass active, +12 is
        # off-limits to the velocity gate and both notes must survive.
        import soundfile as sf

        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        t = np.arange(int(self.SR * 2.0)) / self.SR
        y = np.zeros_like(t)
        burst = (t >= 0.1) & (t < 0.7)
        y[burst] += 0.9 * np.sin(2 * np.pi * 880.0 * t[burst])
        pair = (t >= 1.0) & (t < 1.9)
        y[pair] += 0.01 * (
            np.sin(2 * np.pi * 174.61 * t[pair])  # F3 = MIDI 53
            + 0.4 * np.sin(2 * np.pi * 349.23 * t[pair])  # F4 = MIDI 65
        )
        paths.stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(paths.guitar_stem, y.astype(np.float32), self.SR)
        _write_notes(
            paths,
            [
                {"start": 1.0, "end": 1.9, "pitch": 53, "velocity": 100},
                {"start": 1.02, "end": 1.9, "pitch": 65, "velocity": 100},
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert sorted(n["pitch"] for n in out["notes"]) == [53, 65]
        assert out["overtone_artifacts"] == []

    def test_window_chunking_and_boundary_straddling_cluster(self, tmp_path):
        # An 82 s stem spans two evidence windows (80 s each). Notes land in
        # window 0, in window 1, AND in a cluster straddling the boundary
        # (onsets 79.98 + 80.03, gap 0.05 < ONSET_CLUSTER_GAP) — the
        # straddling cluster is adjudicated from its FIRST onset's window,
        # whose _EVIDENCE_WINDOW_PAD covers past the boundary. The stem is
        # 220 Hz with a 440 Hz partial at 0.05 amplitude, so the claimed
        # A4=69 over A3=57 is a strict phantom (E ratio ~0.035 < 0.20).
        import soundfile as sf

        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        t = np.arange(int(self.SR * 82.0)) / self.SR
        y = 0.5 * (np.sin(2 * np.pi * 220.0 * t) + 0.05 * np.sin(2 * np.pi * 440.0 * t))
        paths.stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(paths.guitar_stem, y.astype(np.float32), self.SR)
        _write_notes(
            paths,
            [
                {"start": 1.0, "end": 1.5, "pitch": 57, "velocity": 100},
                {"start": 79.98, "end": 80.5, "pitch": 57, "velocity": 100},
                {"start": 80.03, "end": 80.5, "pitch": 69, "velocity": 100},
                {"start": 81.0, "end": 81.5, "pitch": 57, "velocity": 100},
            ],
        )
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        # The phantom in the straddling cluster was dropped + recorded; the
        # window-1 note got a measured velocity like everything else.
        assert [n["pitch"] for n in out["notes"]] == [57, 57, 57]
        assert len(out["overtone_artifacts"]) == 1
        assert out["overtone_artifacts"][0]["pitch"] == 69
        assert out["overtone_artifacts"][0]["start"] == 80.03
        for n in out["notes"]:
            assert isinstance(n["velocity"], int)
            assert n["velocity"] > 0
