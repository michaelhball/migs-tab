"""Tests for render.py — time formatting, structure fallback, note filters,
beat-grid slot collisions, and output staleness."""

from __future__ import annotations

import json
import os

from migs_tab.paths import VideoPaths
from migs_tab.render import (
    _DEFAULT_TAB_STRING_LETTERS,
    RenderedSection,
    TuningInfo,
    _apply_articulations_prelayout,
    _apply_cross_instance_support,
    _apply_overrides,
    _apply_verified_chord_shapes,
    _articulation_legend,
    _articulation_note_indices,
    _articulations_in_window,
    _build_cross_instance_support,
    _collect_section_notes,
    _filter_noise,
    _format_full_tab,
    _format_time,
    _load_overrides,
    _nearest_index,
    _outputs_fresh,
    _pick_canonical_instance,
    _refine_tempo_octave,
    _render_section_tab,
    _sections_from_structure,
    _slot_winner,
    _subdivisions_from_beats,
    _tab_string_letters_for_tuning,
    _uniform_beat_grid,
    render,
)


class TestFormatTime:
    def test_seconds_only(self):
        assert _format_time(7.0) == "0:07"

    def test_minutes_and_seconds(self):
        assert _format_time(65.0) == "1:05"

    def test_zero(self):
        assert _format_time(0.0) == "0:00"

    def test_negative_clamped(self):
        assert _format_time(-1.0) == "0:00"


class TestPickCanonicalInstance:
    def test_no_instances(self):
        assert _pick_canonical_instance({"instances": []}) is None

    def test_prefers_slow_walkthrough(self):
        section = {
            "instances": [
                {"start": 0.0, "end": 5.0, "demo_quality": "normal-tempo"},
                {"start": 10.0, "end": 18.0, "demo_quality": "slow-walkthrough"},
            ]
        }
        chosen = _pick_canonical_instance(section)
        assert chosen["demo_quality"] == "slow-walkthrough"

    def test_prefers_longer_at_equal_quality(self):
        section = {
            "instances": [
                {"start": 0.0, "end": 5.0, "demo_quality": "slow-walkthrough"},
                {"start": 10.0, "end": 20.0, "demo_quality": "slow-walkthrough"},
            ]
        }
        chosen = _pick_canonical_instance(section)
        assert chosen["end"] - chosen["start"] == 10.0


class TestSectionsFromStructure:
    def test_synthesizes_per_segment(self, tmp_path):
        path = tmp_path / "structure.json"
        path.write_text(
            json.dumps(
                {
                    "video_id": "abc12345678",
                    "playing_segments": [
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 30.0,
                            "duration": 30.0,
                            "chords": [
                                {"chord": "Am", "start": 0.0, "end": 5.0},
                                {"chord": "G", "start": 5.0, "end": 10.0},
                            ],
                        }
                    ],
                }
            )
        )
        data = _sections_from_structure(path)
        assert data["video_id"] == "abc12345678"
        assert len(data["sections"]) == 1
        assert data["sections"][0]["chord_progression"] == ["Am", "G"]

    def test_skips_segments_with_no_chords(self, tmp_path):
        path = tmp_path / "structure.json"
        path.write_text(
            json.dumps(
                {
                    "playing_segments": [
                        # Span too short, will be filtered out below.
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 1.0,
                            "duration": 1.0,
                            "chords": [{"chord": "C", "start": 0.0, "end": 0.1}],
                        }
                    ]
                }
            )
        )
        data = _sections_from_structure(path)
        assert len(data["sections"]) == 0


class TestFilterNoise:
    def test_drops_short_and_weak(self):
        # Short notes die only when ALSO weak (velocity < 55).
        notes = [
            {"start": 0.0, "end": 0.05, "pitch": 60, "velocity": 40},  # short + weak
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 80},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1

    def test_short_but_loud_staccato_stab_survives(self):
        # LBTD regression: the recovered E-chord stabs are 0.04-0.07s with
        # pseudo-velocities 79-111. A bare duration gate deleted all 12 and
        # the section vanished from the tab. Short + strong must survive.
        notes = [
            {"start": 156.61, "end": 156.66, "pitch": 52, "velocity": 105},
            {"start": 156.62, "end": 156.67, "pitch": 64, "velocity": 80},
            {"start": 156.63, "end": 156.67, "pitch": 59, "velocity": 79},
        ]
        result = _filter_noise(notes)
        assert len(result) == 3

    def test_short_with_missing_velocity_drops(self):
        # No velocity recorded (older frets.json / transient-prone
        # basic-pitch source) → strength unverifiable → conservative drop.
        notes = [
            {"start": 0.0, "end": 0.05, "pitch": 60},
            {"start": 1.0, "end": 1.5, "pitch": 60},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1
        assert result[0]["start"] == 1.0

    def test_short_at_threshold_survives(self):
        # velocity exactly 55 is NOT weak (gate is `< 55`).
        notes = [{"start": 0.0, "end": 0.05, "pitch": 60, "velocity": 55}]
        assert len(_filter_noise(notes)) == 1

    def test_drops_low_velocity(self):
        notes = [
            {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 20},  # too quiet
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 80},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1

    def test_re_pluck_during_sustain_is_kept(self):
        # MT3 reports long ringing durations; re-picking the same string
        # 1s later is a genuine re-articulation, NOT a sustain re-detection.
        # (The old sustain filter dropped these — a 5s strummed-E demo
        # rendered as a single note.)
        notes = [
            {"start": 0.0, "end": 5.0, "pitch": 60, "velocity": 80},
            {"start": 1.0, "end": 5.0, "pitch": 60, "velocity": 80},
            {"start": 2.0, "end": 5.0, "pitch": 60, "velocity": 80},
        ]
        result = _filter_noise(notes)
        assert len(result) == 3

    def test_duplicate_onset_dedupes_to_louder_note(self):
        # Two same-pitch detections of the SAME pluck (onsets 50ms apart,
        # inside _DEDUPE_WINDOW) collapse to one note. The dedupe pass
        # keeps the louder/longer of the pair — not necessarily the first.
        notes = [
            {"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80},
            {"start": 0.05, "end": 0.5, "pitch": 60, "velocity": 70},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1
        assert result[0]["velocity"] == 80

    def test_duplicate_onset_keeps_louder_later_detection(self):
        # When the LATER duplicate is the louder one, it wins.
        notes = [
            {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 60},
            {"start": 0.05, "end": 1.0, "pitch": 60, "velocity": 90},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1
        assert result[0]["velocity"] == 90

    def test_duplicate_window_does_not_cross_pitches(self):
        # Near-simultaneous DIFFERENT pitches (a strummed chord) all survive.
        notes = [
            {"start": 0.0, "end": 1.0, "pitch": 52, "velocity": 80},
            {"start": 0.02, "end": 1.0, "pitch": 57, "velocity": 80},
            {"start": 0.04, "end": 1.0, "pitch": 61, "velocity": 80},
        ]
        result = _filter_noise(notes)
        assert len(result) == 3


class TestNearestIndex:
    def test_finds_exact(self):
        assert _nearest_index([0.0, 1.0, 2.0, 3.0], 2.0) == 2

    def test_finds_closer_below(self):
        assert _nearest_index([0.0, 1.0, 2.0, 3.0], 1.4) == 1

    def test_finds_closer_above(self):
        assert _nearest_index([0.0, 1.0, 2.0, 3.0], 1.6) == 2

    def test_clamps_below(self):
        assert _nearest_index([1.0, 2.0, 3.0], -5.0) == 0

    def test_clamps_above(self):
        assert _nearest_index([1.0, 2.0, 3.0], 100.0) == 2


class TestSubdivisionsFromBeats:
    def test_inserts_intermediate(self):
        beats = [0.0, 1.0, 2.0]
        grid = _subdivisions_from_beats(beats)
        # 2 beats × 2 subs = 4 + 1 = 5 points total (last beat is included).
        assert len(grid) >= 5
        # First and second-to-last should be marked as beat.
        assert grid[0][1] is True

    def test_handles_short_input(self):
        grid = _subdivisions_from_beats([1.0])
        assert len(grid) == 1


class TestUniformBeatGrid:
    def test_at_120_bpm(self):
        # 120 bpm = 0.5s per beat.
        grid = _uniform_beat_grid(0.0, 2.0, 120.0)
        # Should include 0, 0.5, 1.0, 1.5, 2.0 (5 beats).
        assert len(grid) == 5
        assert grid[0] == 0.0
        assert abs(grid[-1] - 2.0) < 0.001


class TestTabStringLetters:
    def test_standard(self):
        # Standard tuning: low E, A, D, G, B, high E.
        letters = _tab_string_letters_for_tuning([40, 45, 50, 55, 59, 64])
        # Top to bottom: e (lowercase) B G D A E.
        assert letters == ["e", "B", "G", "D", "A", "E"]

    def test_drop_d(self):
        letters = _tab_string_letters_for_tuning([38, 45, 50, 55, 59, 64])
        # Bottom string is now D (uppercase).
        assert letters[-1] == "D"
        assert letters[0] == "e"

    def test_double_drop_d(self):
        letters = _tab_string_letters_for_tuning([38, 45, 50, 55, 59, 62])
        # Both top and bottom = D (top lowercase 'd', bottom uppercase 'D').
        assert letters[0] == "d"
        assert letters[-1] == "D"

    def test_invalid_length_falls_back(self):
        letters = _tab_string_letters_for_tuning([40, 45])  # only 2
        assert letters == _DEFAULT_TAB_STRING_LETTERS


class TestTuningInfoFromPaths:
    def test_default_when_no_file(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        info = TuningInfo.from_paths(paths)
        assert info.label == "Standard"
        assert info.capo == 0

    def test_reads_drop_d(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.tuning_json.write_text(
            json.dumps(
                {
                    "label": "Drop D",
                    "capo": 0,
                    "strings_midi": [38, 45, 50, 55, 59, 64],
                    "source": "audio",
                    "confidence": 0.9,
                }
            )
        )
        info = TuningInfo.from_paths(paths)
        assert info.label == "Drop D"
        assert info.strings_midi[0] == 38


class TestApplyOverrides:
    def test_overrides_a_single_note(self):
        notes = [
            {
                "note_index": 0,
                "start": 0.0,
                "end": 1.0,
                "pitch": 57,
                "string": 3,
                "fret": 2,
                "cluster_id": 0,
                "ambiguous": False,
            }
        ]
        overrides = {0: [{"note_index": 0, "string": 2, "fret": 7}]}
        result = _apply_overrides(notes, overrides)
        assert result[0]["string"] == 2
        assert result[0]["fret"] == 7
        assert result[0]["overridden"] is True

    def test_passes_through_when_no_overrides(self):
        notes = [
            {
                "note_index": 0,
                "start": 0.0,
                "end": 1.0,
                "pitch": 57,
                "string": 3,
                "fret": 2,
                "cluster_id": 0,
                "ambiguous": False,
            }
        ]
        result = _apply_overrides(notes, {})
        assert result == notes


class TestLoadOverrides:
    def test_missing_returns_empty(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        assert _load_overrides(paths) == {}

    def test_loads_valid_overrides(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.frets_overrides_json.write_text(
            json.dumps(
                {
                    "overrides": [
                        {
                            "cluster_id": 5,
                            "new_assignments": [{"note_index": 0, "string": 2, "fret": 7}],
                        }
                    ]
                }
            )
        )
        result = _load_overrides(paths)
        assert 5 in result
        assert result[5][0]["fret"] == 7


class TestRefineTempoOctave:
    def test_in_range_unchanged(self):
        tempo, beats = _refine_tempo_octave(100.0, [0.0, 0.6, 1.2, 1.8])
        assert tempo == 100.0

    def test_too_fast_gets_halved(self):
        # Beat-track returned 240 bpm with beats at 0.25s intervals.
        beats = [i * 0.25 for i in range(20)]
        tempo, new_beats = _refine_tempo_octave(240.0, beats)
        # 240 / 2 = 120 (in range).
        assert tempo == 120.0
        # Every other beat survives.
        assert len(new_beats) == 10

    def test_quadruple_too_fast_halves_twice(self):
        beats = [i * 0.1 for i in range(40)]
        tempo, new_beats = _refine_tempo_octave(360.0, beats)
        # 360 → 180 → 90 (in range after two halvings).
        assert tempo == 90.0
        assert len(new_beats) == 10  # 40 / 4

    def test_too_slow_gets_doubled(self):
        # 30 bpm with beats 2s apart — actually 60 bpm.
        beats = [0.0, 2.0, 4.0, 6.0]
        tempo, new_beats = _refine_tempo_octave(30.0, beats)
        # 30 * 2 = 60 (just barely in range — _TEMPO_PLAUSIBLE_MIN = 55).
        assert tempo == 60.0
        # Original 4 beats + 3 inserted midpoints.
        assert len(new_beats) == 7

    def test_unrecoverably_few_beats(self):
        # Only 1 beat — can't double or halve.
        tempo, beats = _refine_tempo_octave(300.0, [1.0])
        assert tempo == 300.0


class TestBuildCrossInstanceSupport:
    def test_skips_sections_with_single_instance(self):
        sections = {
            "sections": [
                {
                    "label": "A",
                    "instances": [{"start": 0.0, "end": 5.0}],
                }
            ]
        }
        result = _build_cross_instance_support(sections, [], [])
        assert "A" not in result

    def test_counts_instances_with_chord_pitch_pair(self):
        sections = {
            "sections": [
                {
                    "label": "intro",
                    "instances": [
                        {"start": 0.0, "end": 5.0},
                        {"start": 10.0, "end": 15.0},
                        {"start": 20.0, "end": 25.0},
                    ],
                }
            ]
        }
        notes = [
            {"start": 1.0, "pitch": 57},  # in inst 0, Am chord
            {"start": 11.0, "pitch": 57},  # in inst 1, Am chord
            # No occurrence in inst 2 — support = 2.
        ]
        chord_spans = [
            (0.0, 5.0, "Am"),
            (10.0, 15.0, "Am"),
            (20.0, 25.0, "Am"),
        ]
        result = _build_cross_instance_support(sections, notes, chord_spans)
        assert result["intro"][("Am", 57)] == 2


def _note(start, end, pitch, string, fret, cluster_id, **extra):
    return {
        "note_index": 0,
        "start": start,
        "end": end,
        "pitch": pitch,
        "string": string,
        "fret": fret,
        "cluster_id": cluster_id,
        "ambiguous": False,
        **extra,
    }


def _standard_tuning() -> TuningInfo:
    return TuningInfo(
        label="Standard",
        capo=0,
        strings_midi=[40, 45, 50, 55, 59, 64],
        source="test",
        confidence=1.0,
        string_letters=list(_DEFAULT_TAB_STRING_LETTERS),
    )


# Mirrors the real Angie chord-shapes-verified.json Am entry: a cowboy Am
# voicing that ALSO carries the fret-5 A4 on the high-e string, i.e. two
# entries on string 5 (midi 64 → open, midi 69 → fret 5).
_AM_VERIFIED = {
    "verified": {
        "Am": {
            "voicing": [
                {"midi_pitch": 45, "string": 1, "fret": 0},
                {"midi_pitch": 52, "string": 2, "fret": 2},
                {"midi_pitch": 57, "string": 3, "fret": 2},
                {"midi_pitch": 60, "string": 4, "fret": 1},
                {"midi_pitch": 64, "string": 5, "fret": 0},
                {"midi_pitch": 69, "string": 5, "fret": 5},
            ],
            "applies_to": "all_spans",
        }
    }
}

_AM_SPAN = [(820.0, 845.0, "Am")]


class TestVerifiedShapeStrumGating:
    """A verified voicing may only relocate notes inside a strummed
    instance of the chord (>= 3 distinct voicing pitches simultaneous, or
    the whole voicing for dyads). Melody/double-stop clusters in a chord
    span are exempt — the override used to hijack them (Angie's fret-5
    double-stop was forced onto open high-e, then collision-dropped)."""

    def test_angie_double_stop_is_exempt(self):
        # The exact flagship failure: A4(69) on e-5 over E4(64) on B-5 at
        # t≈828.4s inside an Am span. Only 2 voicing pitches sound → NOT a
        # strum → both notes keep the fret algorithm's assignment.
        notes = [
            _note(828.42, 828.76, 69, 5, 5, 1123, velocity=98),
            _note(828.45, 828.73, 64, 4, 5, 1123, velocity=109),
        ]
        result = _apply_verified_chord_shapes(notes, _AM_SPAN, _AM_VERIFIED, _standard_tuning())
        assert (result[0]["string"], result[0]["fret"]) == (5, 5)
        assert (result[1]["string"], result[1]["fret"]) == (4, 5)
        assert all("overridden_by" not in n for n in result)

    def test_genuine_five_note_strum_is_remapped(self):
        # A full Am strum (5 voicing pitches at once) inside the span must
        # still be pulled onto the verified voicing.
        notes = [
            _note(830.0, 831.0, 45, 0, 5, 200),  # algorithm guessed E-string 5
            _note(830.01, 831.0, 52, 1, 7, 200),
            _note(830.02, 831.0, 57, 2, 7, 200),
            _note(830.03, 831.0, 60, 3, 5, 200),
            _note(830.04, 831.0, 64, 4, 5, 200),
        ]
        result = _apply_verified_chord_shapes(notes, _AM_SPAN, _AM_VERIFIED, _standard_tuning())
        got = {n["pitch"]: (n["string"], n["fret"]) for n in result}
        assert got == {45: (1, 0), 52: (2, 2), 57: (3, 2), 60: (4, 1), 64: (5, 0)}
        assert all(n["overridden_by"] == "verified-shape:Am" for n in result)

    def test_same_string_targets_within_cluster_are_not_fused(self):
        # A strum cluster containing BOTH midi 64 and 69: the voicing maps
        # both onto string 5, which one strum cannot do — the conflicting
        # pair keeps the algorithm's assignment, the rest is remapped.
        notes = [
            _note(830.0, 831.0, 45, 0, 5, 300),
            _note(830.01, 831.0, 52, 1, 7, 300),
            _note(830.02, 831.0, 57, 2, 7, 300),
            _note(830.03, 831.0, 64, 4, 5, 300),
            _note(830.04, 831.0, 69, 5, 5, 300),
        ]
        result = _apply_verified_chord_shapes(notes, _AM_SPAN, _AM_VERIFIED, _standard_tuning())
        got = {n["pitch"]: (n["string"], n["fret"]) for n in result}
        assert got[45] == (1, 0)
        assert got[52] == (2, 2)
        assert got[57] == (3, 2)
        # The string-5 collision pair is left untouched.
        assert got[64] == (4, 5)
        assert got[69] == (5, 5)

    def test_relocation_onto_unmoved_cluster_mate_is_cancelled(self):
        # The Angie cluster-119 failure shape: the voicing relocates a
        # note onto a string that an UNMOVED cluster-mate (a stray
        # non-voicing pitch) already occupies. The old guard only caught
        # proposal-vs-proposal contests, so the relocated note collided
        # with the unmoved one and _slot_winner silently dropped a real
        # sounded note. The contested remap must be cancelled; the
        # uncontested remaps still apply.
        notes = [
            _note(830.0, 831.0, 45, 0, 5, 500),  # proposes string 1 — contested
            _note(830.01, 831.0, 52, 2, 2, 500),  # already at the verified target
            _note(830.02, 831.0, 57, 3, 2, 500),  # already at the verified target
            _note(830.03, 831.0, 71, 1, 4, 500),  # stray B4, UNMOVED, sits on string 1
        ]
        result = _apply_verified_chord_shapes(notes, _AM_SPAN, _AM_VERIFIED, _standard_tuning())
        got = {n["pitch"]: (n["string"], n["fret"]) for n in result}
        # 45 keeps the algorithm's collision-free assignment instead of
        # being fused onto the stray's string.
        assert got[45] == (0, 5)
        assert got[52] == (2, 2)
        assert got[57] == (3, 2)
        assert got[71] == (1, 4)
        by_pitch = {n["pitch"]: n for n in result}
        assert "overridden_by" not in by_pitch[45]
        assert "overridden_by" not in by_pitch[71]
        # No two cluster notes share a string in the final layout.
        strings = [n["string"] for n in result]
        assert len(strings) == len(set(strings))

    def test_cancelled_remap_revert_cascades_to_fixpoint(self):
        # Cancelling a contested remap reverts that note to its
        # algorithm string, which can newly contest ANOTHER remap's
        # target — the guard must chase the chain to a fixpoint:
        #   71 (unmoved, string 3) contests 57's remap → 57 reverts to
        #   string 2, contesting 52's remap → 52 reverts to string 1,
        #   contesting 45's remap → 45 reverts to string 0 (free).
        notes = [
            _note(830.0, 831.0, 45, 0, 5, 501),
            _note(830.01, 831.0, 52, 1, 7, 501),
            _note(830.02, 831.0, 57, 2, 7, 501),
            _note(830.03, 831.0, 71, 3, 4, 501),  # stray, on 57's target string
        ]
        result = _apply_verified_chord_shapes(notes, _AM_SPAN, _AM_VERIFIED, _standard_tuning())
        got = {n["pitch"]: (n["string"], n["fret"]) for n in result}
        assert got == {45: (0, 5), 52: (1, 7), 57: (2, 7), 71: (3, 4)}
        assert all("overridden_by" not in n for n in result)

    def test_strum_with_extra_non_voicing_pitch_still_qualifies(self):
        # 3 distinct voicing pitches + 1 stray pitch: the members are
        # remapped, the stray keeps its assignment.
        notes = [
            _note(830.0, 831.0, 45, 0, 5, 400),
            _note(830.01, 831.0, 52, 1, 7, 400),
            _note(830.02, 831.0, 57, 2, 7, 400),
            _note(830.03, 831.0, 71, 5, 7, 400),  # B4 — not in the Am voicing
        ]
        result = _apply_verified_chord_shapes(notes, _AM_SPAN, _AM_VERIFIED, _standard_tuning())
        got = {n["pitch"]: (n["string"], n["fret"]) for n in result}
        assert got == {45: (1, 0), 52: (2, 2), 57: (3, 2), 71: (5, 7)}

    def test_dyad_voicing_requires_all_pitches(self):
        # For a 2-pitch (power-chord) voicing the whole voicing must sound.
        verified = {
            "verified": {
                "E5": {
                    "voicing": [
                        {"midi_pitch": 40, "string": 0, "fret": 0},
                        {"midi_pitch": 47, "string": 1, "fret": 2},
                    ],
                    "applies_to": "all_spans",
                }
            }
        }
        spans = [(0.0, 10.0, "E5")]
        # Lone E2 inside the span: melody, not the power chord → exempt.
        solo = [_note(1.0, 2.0, 40, 0, 12, 0)]
        result = _apply_verified_chord_shapes(solo, spans, verified, _standard_tuning())
        assert (result[0]["string"], result[0]["fret"]) == (0, 12)
        # Both pitches together → genuine power chord → remapped.
        both = [
            _note(3.0, 4.0, 40, 0, 12, 1),
            _note(3.02, 4.0, 47, 1, 14, 1),
        ]
        result = _apply_verified_chord_shapes(both, spans, verified, _standard_tuning())
        got = {n["pitch"]: (n["string"], n["fret"]) for n in result}
        assert got == {40: (0, 0), 47: (1, 2)}


class TestSlotWinner:
    """Same-string same-slot keep-rule: prefer (1) the note NOT relocated
    by a chord-shape override, then (2) higher velocity, then (3) earlier
    onset."""

    def test_prefers_note_not_relocated_by_override(self):
        relocated = _note(0.95, 2.0, 64, 5, 0, 0, overridden_by="verified-shape:Am")
        original = _note(1.02, 1.3, 69, 5, 5, 1)
        kept, dropped = _slot_winner(relocated, original)
        assert kept is original
        assert dropped is relocated

    def test_prefers_higher_velocity(self):
        quiet = _note(0.95, 2.0, 64, 5, 0, 0, velocity=60)
        loud = _note(1.02, 1.3, 67, 5, 3, 1, velocity=100)
        kept, dropped = _slot_winner(quiet, loud)
        assert kept is loud

    def test_falls_back_to_earlier_onset(self):
        first = _note(0.95, 2.0, 64, 5, 0, 0, velocity=90)
        second = _note(1.02, 1.3, 67, 5, 3, 1, velocity=90)
        kept, _ = _slot_winner(first, second)
        assert kept is first

    def test_override_preference_beats_velocity(self):
        loud_relocated = _note(0.95, 2.0, 64, 5, 0, 0, velocity=120, overridden_by="x")
        quiet_original = _note(1.02, 1.3, 67, 5, 3, 1, velocity=60)
        kept, _ = _slot_winner(loud_relocated, quiet_original)
        assert kept is quiet_original

    def test_dropped_relocated_note_is_footnoted_with_tag(self):
        # Through _render_section_tab: the relocated open-string note loses
        # and the footnote says so.
        notes = [
            _note(0.95, 2.0, 64, 5, 0, 0, overridden_by="verified-shape:Am"),
            _note(1.02, 1.3, 69, 5, 5, 1),
        ]
        tab, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0]
        )
        assert collisions == 1
        assert len(layout_notes) == 1
        assert "kept fret 5" in layout_notes[0]
        assert "dropped fret 0" in layout_notes[0]
        assert "relocated by a verified-shape override" in layout_notes[0]
        e_line = next(line for line in tab.splitlines() if line.lstrip().startswith("e|"))
        assert "5" in e_line


class TestSlotCollisions:
    # Beat grid for these tests: beats at 0/1/2/3s, 8th-note slots every
    # 0.5s. Clusters that snap to an occupied slot are bumped forward to
    # the next free slot (bounded); residual shared cells are footnoted —
    # they used to merge silently (same-string frets last-write-won).

    def test_later_cluster_shifts_to_adjacent_free_slot(self):
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0),
            _note(0.1, 0.5, 47, 1, 2, 1),  # also snaps to slot 0 → bumped to slot 1
        ]
        tab, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0]
        )
        assert collisions == 1
        assert layout_notes == []  # resolved by shifting, nothing dropped
        a_line = next(line for line in tab.splitlines() if line.lstrip().startswith("A|"))
        # Both frets visible, fret 0 before fret 2 (onset order preserved).
        assert a_line.index("0") < a_line.index("2")

    def test_three_clusters_one_slot_cascade_forward(self):
        # Three clusters all snap to slot 0. The third must NOT fuse back
        # into the original cell — the forward scan walks past the slot the
        # second cluster was bumped to.
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0),
            _note(0.05, 0.5, 47, 1, 2, 1),
            _note(0.1, 0.5, 48, 1, 3, 2),
        ]
        tab, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0]
        )
        assert collisions == 2
        assert layout_notes == []  # all resolved by shifting
        a_line = next(line for line in tab.splitlines() if line.lstrip().startswith("A|"))
        # All three frets visible in onset order.
        assert a_line.index("0") < a_line.index("2") < a_line.index("3")

    def test_unresolvable_collision_keeps_earlier_and_footnotes(self):
        # Grid is [0.0, 0.5, 1.0]; both clusters snap to the LAST slot, so
        # the forward scan finds no free slot to shift into.
        notes = [
            _note(0.95, 2.0, 64, 5, 0, 0),
            _note(1.02, 1.3, 67, 5, 3, 1),  # same string, same slot, no escape
        ]
        tab, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0]
        )
        assert collisions == 1
        assert len(layout_notes) == 1
        assert layout_notes[0].startswith("same-string conflict")
        assert "kept fret 0" in layout_notes[0]
        assert "dropped fret 3" in layout_notes[0]
        e_line = next(line for line in tab.splitlines() if line.lstrip().startswith("e|"))
        assert "3" not in e_line  # the dropped fret is not silently drawn

    def test_cross_string_fusion_is_footnoted(self):
        # Different-string clusters forced into one cell (no free slot)
        # would be drawn as a chord the player never strummed — that must
        # be footnoted, never silent.
        notes = [
            _note(0.95, 2.0, 64, 5, 0, 0),
            _note(1.02, 1.3, 59, 4, 0, 1),  # different string → cell shared
        ]
        _, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0]
        )
        assert collisions == 1
        assert len(layout_notes) == 1
        assert layout_notes[0].startswith("cross-string merge")
        assert "2 separate onsets" in layout_notes[0]

    def test_forward_scan_is_bounded(self):
        # Five clusters snap to slot 0 but only _MAX_SLOT_SHIFT (3) forward
        # slots may absorb bumps; the fifth shares the original cell and is
        # footnoted as a cross-string merge.
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0),
            _note(0.05, 0.4, 47, 1, 2, 1),
            _note(0.1, 0.4, 48, 1, 3, 2),
            _note(0.15, 0.4, 50, 1, 5, 3),
            _note(0.2, 0.4, 52, 2, 2, 4),  # other string; shares slot 0 with cluster 0
        ]
        _, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0, 4.0]
        )
        assert collisions == 4
        assert len(layout_notes) == 1
        assert layout_notes[0].startswith("cross-string merge")

    def test_no_collision_no_count(self):
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0),
            _note(1.0, 1.4, 47, 1, 2, 1),
        ]
        _, collisions, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0]
        )
        assert collisions == 0
        assert layout_notes == []

    def test_footnotes_capped_per_section(self):
        # 11 same-string notes in one cluster → 10 conflicts; footnotes are
        # capped at 8 plus a "+N more" summary so the tab isn't buried.
        notes = [_note(0.0, 0.5, 64 + f, 5, f, 0) for f in range(11)]
        _, _, layout_notes, _ = _render_section_tab(notes, line_width=72, beat_times=[0.0, 1.0])
        assert len(layout_notes) == 9
        assert layout_notes[-1] == "… (+2 more layout conflicts)"


class TestStaleness:
    """render() must re-render when any input is newer than the outputs."""

    def _seed_cache(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path / "cache")
        paths.frets_json.write_text(
            json.dumps(
                {
                    "notes": [
                        _note(0.5, 1.0, 45, 0, 0, 0),
                        _note(1.5, 2.0, 52, 2, 2, 1),
                    ]
                }
            )
        )
        paths.sections_json.write_text(
            json.dumps(
                {
                    "video_id": "aaa11111111",
                    "structural_summary": "one tiny section",
                    "sections": [
                        {
                            "label": "verse",
                            "description": "",
                            "chord_progression": ["Am"],
                            "instances": [
                                {"start": 0.0, "end": 4.0, "demo_quality": "normal-tempo"}
                            ],
                        }
                    ],
                }
            )
        )
        return paths

    def test_rerenders_when_input_newer(self, tmp_path):
        paths = self._seed_cache(tmp_path)
        out_root = tmp_path / "out"
        tab_path = render(paths, output_root=out_root)
        assert tab_path.exists()
        first = tab_path.stat().st_mtime_ns

        # Outputs newer than inputs → second call is a no-op.
        assert render(paths, output_root=out_root) == tab_path
        assert tab_path.stat().st_mtime_ns == first

        # Touch an input into the future → must re-render.
        future = tab_path.stat().st_mtime + 10
        os.utime(paths.frets_json, (future, future))
        render(paths, output_root=out_root)
        assert tab_path.stat().st_mtime_ns != first

    def test_new_optional_input_invalidates(self, tmp_path):
        paths = self._seed_cache(tmp_path)
        out_root = tmp_path / "out"
        tab_path = render(paths, output_root=out_root)
        first = tab_path.stat().st_mtime_ns
        # A tuning.json appearing after the first render counts as a newer input.
        future = tab_path.stat().st_mtime + 10
        paths.tuning_json.write_text(
            json.dumps({"label": "Drop D", "capo": 0, "strings_midi": [38, 45, 50, 55, 59, 64]})
        )
        os.utime(paths.tuning_json, (future, future))
        render(paths, output_root=out_root)
        assert tab_path.stat().st_mtime_ns != first
        assert "Drop D" in tab_path.read_text()

    def test_force_rerenders_even_when_fresh(self, tmp_path):
        paths = self._seed_cache(tmp_path)
        out_root = tmp_path / "out"
        tab_path = render(paths, output_root=out_root)
        md_path = tab_path.with_name("tab.md")
        # Plant a sentinel and date both outputs into the future so the
        # freshness check would normally skip.
        tab_path.write_text("SENTINEL")
        future = paths.frets_json.stat().st_mtime + 100
        os.utime(tab_path, (future, future))
        os.utime(md_path, (future, future))
        assert render(paths, output_root=out_root) == tab_path
        assert tab_path.read_text() == "SENTINEL"  # skipped, still stale
        render(paths, output_root=out_root, force=True)
        assert tab_path.read_text() != "SENTINEL"

    def test_outputs_fresh_requires_both_files(self, tmp_path):
        paths = self._seed_cache(tmp_path)
        out_root = tmp_path / "out"
        tab_path = render(paths, output_root=out_root)
        md_path = tab_path.with_name("tab.md")
        md_path.unlink()
        assert _outputs_fresh(paths, tab_path, md_path) is False


def _line(tab: str, letter: str) -> str:
    """First tab line whose label matches ``letter`` (e.g. 'A|...')."""
    return next(line for line in tab.splitlines() if line.lstrip().startswith(f"{letter}|"))


class TestArticulationPrelayout:
    """Bend-member hiding and harmonic re-stringing happen BEFORE layout,
    so members vanish from note counts and harmonics land on their node
    position regardless of where the Viterbi put them."""

    def test_no_articulations_returns_same_object(self):
        # Byte-stability anchor: an articulation-less frets.json must flow
        # through untouched (not even a copy).
        notes = [_note(0.0, 0.5, 45, 1, 0, 0)]
        assert _apply_articulations_prelayout(notes, []) is notes

    def test_bend_members_hidden(self):
        notes = [
            _note(0.0, 0.6, 58, 3, 3, 0, note_index=10),  # struck A#3, G|3
            _note(0.2, 0.4, 60, 3, 5, 0, note_index=11),  # mid-flight C4 artifact
        ]
        bend = {
            "type": "bend",
            "note_index": 10,
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [11],
            "evidence": {},
        }
        result = _apply_articulations_prelayout(notes, [bend])
        assert [n["note_index"] for n in result] == [10]

    def test_struck_note_never_hidden_even_if_self_listed(self):
        # Defensive: a malformed detection listing the struck note among
        # its own members must not erase the whole bend.
        notes = [_note(0.0, 0.6, 58, 3, 3, 0, note_index=10)]
        bend = {
            "type": "bend",
            "note_index": 10,
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [10],
            "evidence": {},
        }
        result = _apply_articulations_prelayout(notes, [bend])
        assert len(result) == 1

    def test_harmonic_restrung_to_node_position(self):
        # Angie's 12th-fret A-string harmonic: transcribed as A3 (57) and
        # fretted by the Viterbi at G|2 — must move to A|12 and be tagged.
        notes = [_note(1.0, 2.6, 57, 3, 2, 0, note_index=5)]
        harm = {"type": "harmonic", "note_index": 5, "open_string": 1, "node_fret": 12}
        result = _apply_articulations_prelayout(notes, [harm])
        assert result[0]["string"] == 1
        assert result[0]["fret"] == 12
        assert result[0]["natural_harmonic"] is True
        # Original list untouched (copy-on-write).
        assert notes[0]["string"] == 3

    def test_invalid_open_string_skipped(self):
        notes = [_note(1.0, 2.6, 57, 3, 2, 0, note_index=5)]
        harm = {"type": "harmonic", "note_index": 5, "open_string": 9, "node_fret": 12}
        result = _apply_articulations_prelayout(notes, [harm])
        assert result[0]["string"] == 3  # unchanged

    def test_malformed_entries_never_crash(self):
        notes = [_note(0.0, 0.5, 45, 1, 0, 0, note_index=0)]
        artics = [
            {"type": "bend"},  # missing everything
            {"type": "harmonic", "note_index": "x", "open_string": 1, "node_fret": 12},
            {"type": "wat"},
        ]
        assert _apply_articulations_prelayout(notes, artics) is notes


def _pair(typ, from_idx, to_idx, string, from_fret, to_fret):
    return {
        "type": typ,
        "from_note_index": from_idx,
        "note_index": to_idx,
        "string": string,
        "from_fret": from_fret,
        "to_fret": to_fret,
        "evidence": {"onset_ratio": 0.4},
    }


class TestArticulationConnectors:
    # Beat grid [0,1,2,3]: 8th-note slots every 0.5s, indices 0..6; only
    # slot 0 carries a bar marker (bars are 4 beats).

    def test_hammer_connector_adjacent_slots(self):
        # LBTD intro-riff ground truth shape: A|0 then A|3 → 'A|0h3'.
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),
            _note(0.5, 0.9, 48, 1, 3, 1, note_index=1),
        ]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 1, 1, 0, 3)],
        )
        assert "0h3" in _line(tab, "A")
        assert symbols == ["h"]
        assert layout_notes == []

    def test_pull_connector(self):
        # Angie intro motif: e|3 then e|0 → 'e|3p0'.
        notes = [
            _note(0.0, 0.4, 67, 5, 3, 0, note_index=0),
            _note(0.5, 0.9, 64, 5, 0, 1, note_index=1),
        ]
        tab, _, _, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("pull", 0, 1, 5, 3, 0)],
        )
        assert "3p0" in _line(tab, "e")
        assert symbols == ["p"]

    def test_slide_up_connector(self):
        notes = [
            _note(0.0, 0.4, 53, 2, 3, 0, note_index=0),
            _note(0.5, 0.9, 55, 2, 5, 1, note_index=1),
        ]
        tab, _, _, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("slide", 0, 1, 2, 3, 5)],
        )
        assert "3/5" in _line(tab, "D")
        assert symbols == ["/"]

    def test_slide_down_connector(self):
        # Documented convention: descending slides use '\'.
        notes = [
            _note(0.0, 0.4, 55, 2, 5, 0, note_index=0),
            _note(0.5, 0.9, 53, 2, 3, 1, note_index=1),
        ]
        tab, _, _, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("slide", 0, 1, 2, 5, 3)],
        )
        assert "5\\3" in _line(tab, "D")
        assert symbols == ["\\"]

    def test_chained_hammer_pull_shares_middle_note(self):
        # LBTD final-chorus figure D|0h1p0: hammer 0->1 then pull 1->0,
        # the middle note serving as the hammer's destination AND the
        # pull's source. The pull's destination also sits on a beat-
        # marked slot (t=1.0 is beat 2), whose separator is still '-'.
        notes = [
            _note(0.0, 0.4, 50, 2, 0, 0, note_index=0),
            _note(0.5, 0.9, 51, 2, 1, 1, note_index=1),
            _note(1.0, 1.4, 50, 2, 0, 2, note_index=2),
        ]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 1, 2, 0, 1), _pair("pull", 1, 2, 2, 1, 0)],
        )
        assert "0h1p0" in _line(tab, "D")
        assert symbols == ["h", "p"]
        assert layout_notes == []

    def test_connector_into_beat_marked_destination_slot(self):
        # Destination lands exactly ON a beat (slot 2, t=1.0). Beat slots
        # keep the plain '-' separator (only bar slots use '|'), so the
        # connector override applies there like anywhere else.
        notes = [
            _note(0.5, 0.9, 45, 1, 0, 0, note_index=0),  # off-beat slot 1
            _note(1.0, 1.4, 48, 1, 3, 1, note_index=1),  # beat slot 2
        ]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 1, 1, 0, 3)],
        )
        assert "0h3" in _line(tab, "A")
        assert symbols == ["h"]
        assert layout_notes == []

    def test_two_digit_fret_pair_keeps_equal_line_lengths(self):
        # '12/14' — both tokens two digits wide; every string line in the
        # system must stay the same length.
        notes = [
            _note(0.0, 0.4, 57, 1, 12, 0, note_index=0),
            _note(0.5, 0.9, 59, 1, 14, 1, note_index=1),
        ]
        tab, _, _, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("slide", 0, 1, 1, 12, 14)],
        )
        assert "12/14" in _line(tab, "A")
        assert symbols == ["/"]
        for system in tab.split("\n\n"):
            lengths = {len(line) for line in system.splitlines()}
            assert len(lengths) == 1, f"unequal line lengths in system:\n{system}"

    def test_connector_in_widened_slot_padding(self):
        # The destination slot is widened to 2 chars by a 2-digit fret on
        # another string, so the connector lands in the cell padding
        # ('0-h3'), not on the separator — slot widths must not change.
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),
            _note(0.5, 0.9, 48, 1, 3, 1, note_index=1),
            _note(0.5, 0.9, 65, 5, 1, 1, note_index=2),  # placeholder same cluster
        ]
        notes[2]["fret"] = 10  # e|10 widens the slot
        notes[2]["pitch"] = 74
        tab, _, _, _ = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 1, 1, 0, 3)],
        )
        assert "0-h3" in _line(tab, "A")
        assert "10" in _line(tab, "e")

    def test_orphaned_pair_footnoted_not_silent(self):
        # FROM half was dropped by an earlier stage (not in section notes
        # at all): render() only passes in-window articulations, so this
        # IS a real dropped endpoint — footnote, never silence.
        notes = [_note(0.5, 0.9, 48, 1, 3, 1, note_index=1)]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 1, 1, 0, 3)],
        )
        assert "h" not in _line(tab, "A")
        assert symbols == []
        assert len(layout_notes) == 1
        assert "hammer-on 0->3 on A string at +0.5s" in layout_notes[0]
        assert "endpoint note absent from this section's layout" in layout_notes[0]
        assert "connector omitted" in layout_notes[0]

    def test_pair_separated_by_collision_bump_footnotes(self):
        # An interloper cluster occupies the destination slot, so the TO
        # note is bumped one further — pair lands 2 slots apart → footnote,
        # no connector.
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),
            _note(0.45, 0.8, 50, 2, 0, 1, note_index=1),  # interloper, D string
            _note(0.55, 0.9, 48, 1, 3, 2, note_index=2),  # bumped to slot 2
        ]
        tab, collisions, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 2, 1, 0, 3)],
        )
        assert collisions == 1
        assert "0h3" not in _line(tab, "A")
        assert symbols == []
        assert len(layout_notes) == 1
        assert layout_notes[0].startswith("hammer-on 0->3 on A string at +0.0s")
        assert "connector omitted" in layout_notes[0]

    def test_pair_across_bar_line_footnotes(self):
        # Beats [0..5] → bar line before slot 8 (beat 5 at 4.0s). The pair
        # lands in adjacent slots 7 and 8 but in DIFFERENT bars → footnote.
        notes = [
            _note(3.5, 3.9, 45, 1, 0, 0, note_index=0),
            _note(4.0, 4.4, 48, 1, 3, 1, note_index=1),
        ]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            articulations=[_pair("hammer", 0, 1, 1, 0, 3)],
        )
        assert "0h3" not in _line(tab, "A")
        assert "0|3" in _line(tab, "A")  # bar line preserved
        assert symbols == []
        assert len(layout_notes) == 1
        assert "hammer-on 0->3 on A string" in layout_notes[0]

    def test_restrung_half_drops_connector_keeps_footnote(self):
        # A chord-shape override moved the TO note to another string; the
        # articulation's string field no longer matches → footnote only.
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),
            _note(0.5, 0.9, 48, 2, 3, 1, note_index=1),  # re-strung to D
        ]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("hammer", 0, 1, 1, 0, 3)],
        )
        assert "h" not in _line(tab, "A")
        assert symbols == []
        assert len(layout_notes) == 1
        assert "pair re-strung by overrides" in layout_notes[0]

    def test_pair_referencing_hidden_bend_member_footnotes(self):
        # A hammer whose destination is a bend member: the member is hidden
        # pre-layout, so the pair half is missing → footnoted (invariant:
        # mark or footnote, never silence), while the bend itself renders
        # with its target-pitch footnote.
        notes = [
            _note(0.0, 0.6, 58, 3, 3, 0, note_index=10),
            _note(0.2, 0.4, 60, 3, 5, 0, note_index=11),  # member
        ]
        bend = {
            "type": "bend",
            "note_index": 10,
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [11],
            "evidence": {},
        }
        artics = [bend, _pair("hammer", 10, 11, 3, 3, 5)]
        kept = _apply_articulations_prelayout(notes, artics)
        tab, _, layout_notes, symbols = _render_section_tab(
            kept,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=artics,
        )
        assert "3b5" in _line(tab, "G")
        assert symbols == ["b"]
        assert len(layout_notes) == 2
        assert layout_notes[0] == "b5 = bend toward C4"  # A#3(58) + 2 = C4
        assert "hammer-on 3->5 on G string" in layout_notes[1]
        assert "connector omitted" in layout_notes[1]

    def test_bend_on_restrung_note_footnoted_not_drawn(self):
        # The struck note no longer sits at the measured (string, fret) —
        # an override moved it — so the 'b' mark is dropped + footnoted.
        notes = [_note(0.0, 0.6, 58, 4, 6, 0, note_index=10)]  # moved to B|6
        bend = {
            "type": "bend",
            "note_index": 10,
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [],
            "evidence": {},
        }
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[bend],
        )
        assert "b" not in _line(tab, "B")
        assert symbols == []
        assert len(layout_notes) == 1
        assert "bend 3->5" in layout_notes[0]
        assert "bend mark dropped" in layout_notes[0]

    def test_malformed_bend_missing_string_no_token_no_footnote(self):
        # Entry missing "string" on a note that IS at the measured G|3:
        # the token path never drew 'b', so the footnote path must not
        # blame "overrides" that never ran — same field set both paths.
        notes = [_note(0.0, 0.6, 58, 3, 3, 0, note_index=10)]
        bend = {
            "type": "bend",
            "note_index": 10,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [],
            "evidence": {},
        }
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[bend],
        )
        assert "b" not in _line(tab, "G")
        assert symbols == []
        assert layout_notes == []

    def test_string_typed_note_index_token_and_footnote_agree(self):
        # A JSON-stringly note_index ("10") is int-coerced once when
        # bend_by_idx is built, so the token path and the footnote path
        # can never disagree on whether the entry exists: token drawn,
        # no footnote.
        notes = [_note(0.0, 0.6, 58, 3, 3, 0, note_index=10)]
        bend = {
            "type": "bend",
            "note_index": "10",
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [],
            "evidence": {},
        }
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[bend],
        )
        assert "3b5" in _line(tab, "G")
        assert symbols == ["b"]
        assert layout_notes == ["b5 = bend toward C4"]

    def test_no_articulations_identical_output(self):
        notes = [
            _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),
            _note(0.5, 0.9, 48, 1, 3, 1, note_index=1),
        ]
        base = _render_section_tab(notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0])
        with_none = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0], articulations=None
        )
        with_empty = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0], articulations=[]
        )
        assert base == with_none == with_empty
        assert base[3] == []


def _five_type_fixture():
    """One section exercising all five articulation types at once."""
    notes = [
        _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),  # A|0 ┐ hammer
        _note(0.5, 0.9, 48, 1, 3, 1, note_index=1),  # A|3 ┘
        _note(1.0, 1.4, 67, 5, 3, 2, note_index=2),  # e|3 ┐ pull
        _note(1.5, 1.9, 64, 5, 0, 3, note_index=3),  # e|0 ┘
        _note(2.0, 2.4, 53, 2, 3, 4, note_index=4),  # D|3 ┐ slide up
        _note(2.5, 2.9, 55, 2, 5, 5, note_index=5),  # D|5 ┘
        _note(3.0, 3.6, 58, 3, 3, 6, note_index=6),  # G|3 bend struck (A#3)
        _note(3.1, 3.3, 60, 3, 5, 6, note_index=7),  # mid-flight C4 artifact
        _note(4.0, 5.6, 57, 3, 2, 7, note_index=8),  # A3 at G|2 → harmonic A|<12>
    ]
    artics = [
        _pair("hammer", 0, 1, 1, 0, 3),
        _pair("pull", 2, 3, 5, 3, 0),
        _pair("slide", 4, 5, 2, 3, 5),
        {
            "type": "bend",
            "note_index": 6,
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [7],
            "evidence": {"cqt_trajectory": "A#3->C4"},
        },
        {
            "type": "harmonic",
            "note_index": 8,
            "open_string": 1,
            "node_fret": 12,
            "evidence": {"octave_pair": True},
        },
    ]
    return notes, artics


class TestAllFiveTypes:
    def _render(self):
        notes, artics = _five_type_fixture()
        kept = _apply_articulations_prelayout(notes, artics)
        return _render_section_tab(
            kept,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            articulations=artics,
        )

    def test_all_tokens_present(self):
        tab, _, layout_notes, symbols = self._render()
        assert "0h3" in _line(tab, "A")
        assert "3p0" in _line(tab, "e")
        assert "3/5" in _line(tab, "D")
        assert "3b5" in _line(tab, "G")
        assert "<12>" in _line(tab, "A")
        assert symbols == ["/", "<n>", "b", "h", "p"]
        # The only footnote is the drawn bend's target pitch.
        assert layout_notes == ["b5 = bend toward C4"]

    def test_equal_line_lengths_in_every_system(self):
        # THE ASCII-tab invariant: mixed-width tokens (<12>, 3b5) must
        # never skew one string line against the others.
        tab, _, _, _ = self._render()
        for system in tab.split("\n\n"):
            lengths = {len(line) for line in system.splitlines()}
            assert len(lengths) == 1, f"unequal line lengths in system:\n{system}"

    def test_member_note_excluded_from_layout(self):
        notes, artics = _five_type_fixture()
        kept = _apply_articulations_prelayout(notes, artics)
        assert len(kept) == len(notes) - 1
        tab, _, _, _ = self._render()
        # The artifact's fret-5 G-string token must not appear anywhere:
        # the only G-line tokens are the bend's '3b5'.
        g_line = _line(tab, "G")
        assert g_line.count("5") == 1  # the '5' inside '3b5'


class TestArticulationLegend:
    def _sec(self, symbols):
        return RenderedSection(
            label="x",
            description="",
            canonical_start=0.0,
            canonical_end=1.0,
            chord_progression=[],
            cluster_count=0,
            note_count=0,
            ascii_tab="",
            tempo_bpm=90.0,
            artic_symbols=symbols,
        )

    def test_empty_when_no_symbols(self):
        assert _articulation_legend([self._sec([])]) == ""

    def test_lists_only_used_symbols_in_fixed_order(self):
        legend = _articulation_legend([self._sec(["h", "b"]), self._sec(["<n>"])])
        assert legend == "Articulations: h hammer-on · b bend · <n> natural harmonic"

    def test_slide_directions_named_separately(self):
        legend = _articulation_legend([self._sec(["/", "\\"])])
        assert legend == "Articulations: / slide up · \\ slide down"

    def test_markdown_variant_backticks_symbols(self):
        # The tab.md legend sits OUTSIDE the code fences, where a bare
        # '<n>' is a raw-HTML open tag (GitHub swallows it) and '\' is an
        # escape — code spans keep both visible.
        legend = _articulation_legend([self._sec(["h", "\\", "<n>"])], markdown=True)
        assert legend == ("Articulations: `h` hammer-on · `\\` slide down · `<n>` natural harmonic")

    def test_full_tab_carries_legend_line(self):
        sections_data = {"video_id": "test", "sections": []}
        tuning = _standard_tuning()
        text = _format_full_tab(sections_data, [self._sec(["h", "p"])], tuning)
        assert "Articulations: h hammer-on · p pull-off" in text
        # And absent when nothing was drawn (byte-stability).
        text_plain = _format_full_tab(sections_data, [self._sec([])], tuning)
        assert "Articulations:" not in text_plain


class TestRenderWithArticulations:
    """End-to-end through render(): legend, member-aware note counts, and
    byte-stability when frets.json has no articulations."""

    def _seed(self, tmp_path, articulations=None):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path / "cache")
        # No guitar stem → _detect_beats falls back to a uniform 90 bpm
        # grid (slots every 1/3 s).
        frets = {
            "notes": [
                _note(0.0, 0.4, 45, 1, 0, 0, note_index=0),
                _note(0.34, 0.7, 48, 1, 3, 1, note_index=1),
                _note(1.0, 1.5, 58, 3, 3, 2, note_index=2),
                _note(1.05, 1.2, 60, 3, 5, 2, note_index=3),  # bend member
                _note(2.0, 3.5, 57, 3, 2, 3, note_index=4),  # harmonic source
            ]
        }
        if articulations is not None:
            frets["articulations"] = articulations
        paths.frets_json.write_text(json.dumps(frets))
        paths.sections_json.write_text(
            json.dumps(
                {
                    "video_id": "aaa11111111",
                    "structural_summary": "one tiny section",
                    "sections": [
                        {
                            "label": "riff",
                            "description": "",
                            "chord_progression": ["A"],
                            "instances": [
                                {"start": 0.0, "end": 4.0, "demo_quality": "normal-tempo"}
                            ],
                        }
                    ],
                }
            )
        )
        return paths

    def _artics(self):
        return [
            _pair("hammer", 0, 1, 1, 0, 3),
            {
                "type": "bend",
                "note_index": 2,
                "string": 3,
                "fret": 3,
                "target_semitones": 2,
                "member_note_indices": [3],
                "evidence": {},
            },
            {
                "type": "harmonic",
                "note_index": 4,
                "open_string": 1,
                "node_fret": 12,
                "evidence": {},
            },
        ]

    def test_tab_has_tokens_legend_and_member_aware_count(self, tmp_path):
        paths = self._seed(tmp_path, articulations=self._artics())
        tab_path = render(paths, output_root=tmp_path / "out")
        text = tab_path.read_text()
        assert "0h3" in text
        assert "3b5" in text
        assert "<12>" in text
        assert "Articulations: h hammer-on · b bend · <n> natural harmonic" in text
        # 5 notes in frets.json, 1 hidden bend member → 4 in the count.
        assert "(4 notes" in text
        # Markdown mirrors the legend, with symbols backtick-quoted so
        # '<n>' survives CommonMark's raw-HTML parsing outside the fences.
        md = tab_path.with_name("tab.md").read_text()
        assert "Articulations: `h` hammer-on · `b` bend · `<n>` natural harmonic" in md
        assert "Articulations: h hammer-on" not in md

    def test_absent_articulations_is_byte_stable(self, tmp_path):
        # No "articulations" key vs explicit empty list → identical bytes.
        p1 = self._seed(tmp_path / "a")
        t1 = render(p1, output_root=tmp_path / "a" / "out").read_text()
        p2 = self._seed(tmp_path / "b", articulations=[])
        t2 = render(p2, output_root=tmp_path / "b" / "out").read_text()
        assert t1 == t2
        assert "Articulations:" not in t1


class TestArticulationEndpointProtection:
    """GAP 2: articulation endpoints are audio-verified legato events and
    quiet BY NATURE — the velocity floor and the cross-instance loudness
    vote must not eat them (the detection's audio evidence outranks a
    velocity heuristic)."""

    def test_quiet_endpoint_survives_velocity_floor(self):
        # Angie motif pull A|3->0 at t≈131.9s: the vel-22 from-note fell
        # to _MIN_NOTE_VELOCITY=35 and the pull lost its origin.
        notes = [
            {"note_index": 7, "start": 131.87, "end": 132.1, "pitch": 48, "velocity": 22},
            {"note_index": 8, "start": 132.2, "end": 132.5, "pitch": 45, "velocity": 80},
        ]
        assert len(_filter_noise(notes)) == 1
        assert len(_filter_noise(notes, protected={7, 8})) == 2

    def test_protection_does_not_revive_short_and_weak(self):
        # Only the quiet floor is exempted — a short AND weak note still
        # drops (its existence, not just its strength, is in doubt).
        notes = [{"note_index": 7, "start": 0.0, "end": 0.05, "pitch": 60, "velocity": 22}]
        assert _filter_noise(notes, protected={7}) == []

    def test_cross_instance_vote_spares_protected_endpoints(self):
        # LBTD [intro_riff_demo] 0h1p0: the D|0 origins (vel 63/42) miss
        # the >=75 loudness escape and the figure rendered a lone '1'.
        section_notes = [
            {"note_index": 126, "start": 70.38, "pitch": 50, "velocity": 63},
            {"note_index": 127, "start": 70.68, "pitch": 51, "velocity": 76},
            {"note_index": 128, "start": 70.83, "pitch": 50, "velocity": 42},
        ]
        spans = [(61.6, 73.0, "A")]
        supports = {("A", 51): 2}  # only the middle note confirmed by the other take
        kept = _apply_cross_instance_support(
            section_notes, spans, supports, cross_support_min=1, instance_count=2
        )
        assert [n["note_index"] for n in kept] == [127]
        kept = _apply_cross_instance_support(
            section_notes,
            spans,
            supports,
            cross_support_min=1,
            instance_count=2,
            protected={126, 127, 128},
        )
        assert [n["note_index"] for n in kept] == [126, 127, 128]

    def test_protected_set_covers_endpoints_not_members(self):
        arts = [
            _pair("hammer", 0, 1, 1, 0, 3),
            {
                "type": "bend",
                "note_index": 6,
                "string": 3,
                "fret": 3,
                "target_semitones": 2,
                "member_note_indices": [7],
                "evidence": {},
            },
            {"type": "harmonic", "note_index": 8, "open_string": 1, "node_fret": 12},
            {"type": "bend"},  # malformed — contributes nothing
        ]
        assert _articulation_note_indices(arts) == {0, 1, 6, 8}


class TestWindowEdgeAnchor:
    """GAP 1: an articulation endpoint within _ARTIC_EDGE_EPSILON before
    the section window is anchored into the section when its partner is
    inside; ordinary out-of-window notes stay excluded. LBTD's A|0h3
    from-note (t=0.39s) misses [song_full_demo] (starts 0.4s) by 0.01s."""

    def _notes(self):
        return [
            _note(0.39, 0.62, 45, 1, 0, 0, note_index=0, velocity=116),  # 0.01s early
            _note(0.64, 0.83, 48, 1, 3, 1, note_index=1, velocity=72),
            _note(0.30, 0.50, 52, 2, 2, 2, note_index=9, velocity=90),  # ordinary early
        ]

    def test_endpoint_just_before_window_is_anchored(self):
        arts = [_pair("hammer", 0, 1, 1, 0, 3)]
        got = _collect_section_notes(self._notes(), arts, 0.4, 20.8)
        assert [n["note_index"] for n in got] == [0, 1]

    def test_ordinary_early_note_stays_excluded(self):
        # note_index 9 starts inside the epsilon band (0.30 >= 0.25) but
        # is no articulation endpoint → excluded; without articulations
        # the collection is the plain window filter.
        got = _collect_section_notes(self._notes(), [], 0.4, 20.8)
        assert [n["note_index"] for n in got] == [1]

    def test_endpoint_beyond_epsilon_not_anchored(self):
        notes = [
            _note(0.2, 0.45, 45, 1, 0, 0, note_index=0),  # 0.2 < 0.4 - 0.15
            _note(0.64, 0.83, 48, 1, 3, 1, note_index=1),
        ]
        arts = [_pair("hammer", 0, 1, 1, 0, 3)]
        got = _collect_section_notes(notes, arts, 0.4, 20.8)
        assert [n["note_index"] for n in got] == [1]

    def test_partner_must_be_inside_window(self):
        # The TO note falls after the window end, so nothing anchors the
        # FROM note — only the unrelated in-window note is collected.
        notes = self._notes() + [_note(0.5, 0.55, 57, 3, 2, 3, note_index=3)]
        arts = [_pair("hammer", 0, 1, 1, 0, 3)]
        got = _collect_section_notes(notes, arts, 0.4, 0.6)
        assert [n["note_index"] for n in got] == [3]

    def test_anchor_refused_when_another_window_owns_the_note(self):
        # Temporally adjacent canonical windows (prev end == next start):
        # the would-be anchor at 0.39 lies inside another section's
        # [0.0, 0.4) window, which owns and draws it — anchoring it here
        # too would render and count the note twice.
        arts = [_pair("hammer", 0, 1, 1, 0, 3)]
        got = _collect_section_notes(self._notes(), arts, 0.4, 20.8, other_windows=[(0.0, 0.4)])
        assert [n["note_index"] for n in got] == [1]

    def test_anchor_kept_when_other_windows_are_elsewhere(self):
        # Other sections' windows that don't contain the note leave the
        # epsilon anchor intact.
        arts = [_pair("hammer", 0, 1, 1, 0, 3)]
        got = _collect_section_notes(self._notes(), arts, 0.4, 20.8, other_windows=[(30.0, 40.0)])
        assert [n["note_index"] for n in got] == [0, 1]

    def test_render_anchors_edge_endpoint_end_to_end(self, tmp_path):
        # Through render(): the from-note 0.01s before the window renders
        # its 0h3 and the section header counts it.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path / "cache")
        paths.frets_json.write_text(
            json.dumps(
                {
                    "notes": [
                        _note(0.39, 0.62, 45, 1, 0, 0, note_index=0, velocity=116),
                        _note(0.64, 0.83, 48, 1, 3, 1, note_index=1, velocity=72),
                    ],
                    "articulations": [_pair("hammer", 0, 1, 1, 0, 3)],
                }
            )
        )
        paths.sections_json.write_text(
            json.dumps(
                {
                    "video_id": "aaa11111111",
                    "structural_summary": "",
                    "sections": [
                        {
                            "label": "song_full_demo",
                            "description": "",
                            "chord_progression": ["A"],
                            "instances": [{"start": 0.4, "end": 4.0, "demo_quality": "full-tempo"}],
                        }
                    ],
                }
            )
        )
        text = render(paths, output_root=tmp_path / "out").read_text()
        assert "0h3" in text
        assert "(2 notes" in text

    def test_render_adjacent_windows_never_draw_a_note_twice(self, tmp_path):
        # Two sections with temporally adjacent canonical windows and a
        # hammer pair straddling the boundary: the from-note (t=2.3, fret
        # 7) belongs to [verse] and must render exactly once — [chorus]
        # must not epsilon-anchor it on top. Neither section can draw the
        # connector, so BOTH footnote it with the window-edge-aware
        # wording (no silent drop, no spurious "dropped" blame).
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path / "cache")
        paths.frets_json.write_text(
            json.dumps(
                {
                    "notes": [
                        _note(2.3, 2.45, 52, 1, 7, 0, note_index=0, velocity=100),
                        _note(2.5, 2.7, 54, 1, 9, 1, note_index=1, velocity=70),
                    ],
                    "articulations": [_pair("hammer", 0, 1, 1, 7, 9)],
                }
            )
        )
        paths.sections_json.write_text(
            json.dumps(
                {
                    "video_id": "aaa11111111",
                    "structural_summary": "",
                    "sections": [
                        {
                            "label": "verse",
                            "description": "",
                            "chord_progression": ["A"],
                            "instances": [{"start": 0.4, "end": 2.4, "demo_quality": "full-tempo"}],
                        },
                        {
                            "label": "chorus",
                            "description": "",
                            "chord_progression": ["D"],
                            "instances": [{"start": 2.4, "end": 4.4, "demo_quality": "full-tempo"}],
                        },
                    ],
                }
            )
        )
        text = render(paths, output_root=tmp_path / "out").read_text()
        a_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("A|")]
        assert sum(ln.count("7") for ln in a_lines) == 1  # from-note drawn once
        assert sum(ln.count("9") for ln in a_lines) == 1  # to-note drawn once
        assert "7h9" not in text  # pair straddles the windows; no connector
        assert text.count("endpoint note absent from this section's layout") == 2


class TestNoSilentDropInvariant:
    """GAP 3a: every in-window articulation either renders its mark or
    leaves a footnote — no silent connector drops."""

    @staticmethod
    def _mark_count(tab: str) -> int:
        # Tab-body lines contain only '-', '|', digits and articulation
        # chars after the string label, so counting h/p/b//\\ chars plus
        # '<' tokens counts exactly the drawn marks. Lines without '|'
        # (e.g. the blank line between wrapped systems) are skipped.
        count = 0
        for line in tab.splitlines():
            _, sep, body = line.partition("|")
            if not sep:
                continue
            count += sum(body.count(c) for c in "hpb/\\")
            count += body.count("<")
        return count

    @staticmethod
    def _omission_count(layout_notes: list[str]) -> int:
        return sum(
            1
            for s in layout_notes
            if "connector omitted" in s or "bend mark" in s or "mark omitted" in s
        )

    def test_marks_plus_footnotes_cover_every_articulation(self):
        notes, artics = _five_type_fixture()
        # Simulate earlier-stage drops: the hammer's FROM note (idx 0),
        # the slide's TO note (idx 5) and the harmonic note (idx 8).
        kept = [
            n
            for n in _apply_articulations_prelayout(notes, artics)
            if n["note_index"] not in {0, 5, 8}
        ]
        tab, _, layout_notes, _ = _render_section_tab(
            kept,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            articulations=artics,
        )
        # Drawn: pull 'p' + bend 'b' = 2 marks; footnoted: hammer, slide,
        # harmonic = 3 omissions → all 5 articulations accounted for.
        assert self._mark_count(tab) == 2
        assert self._omission_count(layout_notes) == 3
        assert self._mark_count(tab) + self._omission_count(layout_notes) == len(artics)

    def test_articulation_footnotes_exempt_from_layout_cap(self):
        # CAP-PROOF: 11 same-string notes in one cluster → 10 collision
        # footnotes (over the cap of 8) PLUS an orphaned pair whose
        # endpoints never reached layout. The articulation footnote must
        # come FIRST and survive the cap; the "+N more" line collapses
        # only the collision footnotes.
        notes = [_note(0.0, 0.5, 64 + f, 5, f, 0) for f in range(11)]
        _, _, layout_notes, _ = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0],
            articulations=[_pair("pull", 90, 91, 5, 3, 0)],
        )
        assert layout_notes[0] == (
            "pull-off 3->0 on e string (endpoint note absent from this "
            "section's layout — dropped earlier or outside this window; "
            "connector omitted)"
        )
        assert layout_notes[-1] == "… (+2 more layout conflicts)"
        assert len(layout_notes) == 10  # 1 artic + 8 collisions + '+N more'
        assert self._omission_count(layout_notes) == 1

    def test_both_endpoints_dropped_footnoted(self):
        # Both halves gone (e.g. short-and-weak gate + dedupe) — the
        # footnote has no timestamp but still names the figure.
        notes = [_note(2.0, 2.4, 52, 2, 2, 9, note_index=9)]
        _, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[_pair("pull", 0, 1, 5, 3, 0)],
        )
        assert symbols == []
        assert layout_notes == [
            "pull-off 3->0 on e string (endpoint note absent from this "
            "section's layout — dropped earlier or outside this window; "
            "connector omitted)"
        ]

    def test_bend_struck_note_dropped_footnoted(self):
        notes = [_note(2.0, 2.4, 52, 2, 2, 9, note_index=9)]
        bend = {
            "type": "bend",
            "note_index": 10,
            "string": 3,
            "fret": 3,
            "target_semitones": 2,
            "member_note_indices": [],
            "evidence": {},
        }
        _, _, layout_notes, symbols = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0], articulations=[bend]
        )
        assert symbols == []
        assert layout_notes == ["bend 3->5 (struck note dropped before layout; bend mark omitted)"]

    def test_harmonic_note_dropped_footnoted(self):
        notes = [_note(2.0, 2.4, 52, 2, 2, 9, note_index=9)]
        harm = {"type": "harmonic", "note_index": 5, "open_string": 1, "node_fret": 12}
        _, _, layout_notes, symbols = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0], articulations=[harm]
        )
        assert symbols == []
        assert layout_notes == [
            "natural harmonic <12> on A string (note dropped before layout; mark omitted)"
        ]

    def test_articulations_windowed_per_section(self):
        # The invariant relies on render() handing each section ONLY its
        # own articulations: an entry counts as in-window when any
        # referenced RAW note starts inside [start, end).
        arts = [
            _pair("hammer", 0, 1, 1, 0, 3),  # notes at 0.39 / 0.64
            _pair("pull", 4, 5, 5, 3, 0),  # notes at 10.1 / 10.4
            {
                "type": "bend",
                "note_index": 7,
                "string": 3,
                "fret": 3,
                "target_semitones": 2,
                "member_note_indices": [],
                "evidence": {},
            },  # note at 20.0
        ]
        starts = {0: 0.39, 1: 0.64, 4: 10.1, 5: 10.4, 7: 20.0}
        assert _articulations_in_window(arts, starts, 0.4, 5.0) == [arts[0]]
        assert _articulations_in_window(arts, starts, 5.0, 15.0) == [arts[1]]
        assert _articulations_in_window(arts, starts, 15.0, 25.0) == [arts[2]]


class TestBendTargetFootnote:
    """GAP 3b: a drawn bend footnotes the measured target pitch — the
    '3b4' token names a target FRET, not what the bend should sound
    like."""

    def _bend(self, semitones=1):
        return {
            "type": "bend",
            "note_index": 10,
            "string": 3,
            "fret": 3,
            "target_semitones": semitones,
            "member_note_indices": [],
            "evidence": {},
        }

    def test_drawn_bend_names_target_pitch(self):
        # A#3 (58) at G|3 bent +1 semitone → 'b4 = bend toward B3'.
        notes = [_note(0.0, 0.6, 58, 3, 3, 0, note_index=10)]
        tab, _, layout_notes, symbols = _render_section_tab(
            notes,
            line_width=72,
            beat_times=[0.0, 1.0, 2.0, 3.0],
            articulations=[self._bend()],
        )
        assert "3b4" in _line(tab, "G")
        assert symbols == ["b"]
        assert layout_notes == ["b4 = bend toward B3"]

    def test_identical_targets_footnoted_once(self):
        notes = [
            _note(0.0, 0.4, 58, 3, 3, 0, note_index=10),
            _note(1.0, 1.4, 58, 3, 3, 1, note_index=12),
        ]
        bends = [self._bend(), dict(self._bend(), note_index=12)]
        _, _, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0], articulations=bends
        )
        assert layout_notes == ["b4 = bend toward B3"]

    def test_dropped_bend_mark_has_no_target_footnote(self):
        # Re-strung struck note: 'bend mark dropped' footnote only.
        notes = [_note(0.0, 0.6, 58, 4, 6, 0, note_index=10)]
        _, _, layout_notes, _ = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0], articulations=[self._bend()]
        )
        assert len(layout_notes) == 1
        assert "bend mark dropped" in layout_notes[0]
        assert "bend toward" not in layout_notes[0]
