"""Tests for render.py — time formatting, structure fallback, note filters,
beat-grid slot collisions, and output staleness."""

from __future__ import annotations

import json
import os

from migs_tab.paths import VideoPaths
from migs_tab.render import (
    _DEFAULT_TAB_STRING_LETTERS,
    TuningInfo,
    _apply_overrides,
    _apply_verified_chord_shapes,
    _build_cross_instance_support,
    _filter_noise,
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
        tab, collisions, layout_notes = _render_section_tab(
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
        tab, collisions, layout_notes = _render_section_tab(
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
        tab, collisions, layout_notes = _render_section_tab(
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
        tab, collisions, layout_notes = _render_section_tab(
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
        _, collisions, layout_notes = _render_section_tab(
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
        _, collisions, layout_notes = _render_section_tab(
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
        _, collisions, layout_notes = _render_section_tab(
            notes, line_width=72, beat_times=[0.0, 1.0, 2.0, 3.0]
        )
        assert collisions == 0
        assert layout_notes == []

    def test_footnotes_capped_per_section(self):
        # 11 same-string notes in one cluster → 10 conflicts; footnotes are
        # capped at 8 plus a "+N more" summary so the tab isn't buried.
        notes = [_note(0.0, 0.5, 64 + f, 5, f, 0) for f in range(11)]
        _, _, layout_notes = _render_section_tab(notes, line_width=72, beat_times=[0.0, 1.0])
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
