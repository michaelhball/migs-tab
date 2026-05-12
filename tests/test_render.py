"""Tests for render.py — time formatting, structure fallback, note filters."""

from __future__ import annotations

import json

from migs_tab.paths import VideoPaths
from migs_tab.render import (
    _DEFAULT_TAB_STRING_LETTERS,
    TuningInfo,
    _apply_overrides,
    _build_cross_instance_support,
    _filter_noise,
    _format_time,
    _load_overrides,
    _nearest_index,
    _pick_canonical_instance,
    _sections_from_structure,
    _subdivisions_from_beats,
    _tab_string_letters_for_tuning,
    _uniform_beat_grid,
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
    def test_drops_very_short(self):
        notes = [
            {"start": 0.0, "end": 0.05, "pitch": 60, "velocity": 80},  # too short
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 80},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1

    def test_drops_low_velocity(self):
        notes = [
            {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 20},  # too quiet
            {"start": 1.0, "end": 1.5, "pitch": 60, "velocity": 80},
        ]
        result = _filter_noise(notes)
        assert len(result) == 1

    def test_sustain_detection_drops_re_attack(self):
        # Same pitch, second onset inside first's still-ringing window.
        notes = [
            {"start": 0.0, "end": 2.0, "pitch": 60, "velocity": 80},
            {"start": 1.0, "end": 2.0, "pitch": 60, "velocity": 80},  # mid-sustain
        ]
        result = _filter_noise(notes)
        assert len(result) == 1
        assert result[0]["start"] == 0.0


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
