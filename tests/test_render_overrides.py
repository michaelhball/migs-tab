"""Tests for the vision-verified chord-shape override flow in render.py."""

from __future__ import annotations

import json

from migs_tab.paths import VideoPaths
from migs_tab.render import (
    _apply_verified_chord_shapes,
    _in_any_time_range,
    _load_verified_chord_shapes,
)


class TestInAnyTimeRange:
    def test_inside_range(self):
        assert _in_any_time_range([{"start": 1.0, "end": 5.0}], 3.0) is True

    def test_outside_range(self):
        assert _in_any_time_range([{"start": 1.0, "end": 5.0}], 10.0) is False

    def test_at_start_inclusive(self):
        assert _in_any_time_range([{"start": 1.0, "end": 5.0}], 1.0) is True

    def test_at_end_exclusive(self):
        assert _in_any_time_range([{"start": 1.0, "end": 5.0}], 5.0) is False

    def test_multiple_ranges_any_match(self):
        ranges = [{"start": 0.0, "end": 2.0}, {"start": 10.0, "end": 12.0}]
        assert _in_any_time_range(ranges, 11.0) is True

    def test_handles_garbage_entries(self):
        ranges = [{"start": "bad"}, {"start": 1.0, "end": 5.0}]
        assert _in_any_time_range(ranges, 3.0) is True

    def test_empty_returns_false(self):
        assert _in_any_time_range([], 5.0) is False


class TestLoadVerifiedChordShapes:
    def test_missing_returns_empty(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        assert _load_verified_chord_shapes(paths) == {}

    def test_loads_valid_json(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.chord_shapes_verified_json.write_text(
            json.dumps(
                {
                    "video_id": "aaa11111111",
                    "verified": {
                        "Am": {
                            "voicing": [
                                {"midi_pitch": 45, "string": 1, "fret": 0},
                            ],
                            "applies_to": "all_spans",
                        }
                    },
                }
            )
        )
        result = _load_verified_chord_shapes(paths)
        assert "verified" in result
        assert "Am" in result["verified"]

    def test_handles_invalid_json(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.chord_shapes_verified_json.write_text("not json")
        assert _load_verified_chord_shapes(paths) == {}


class TestApplyVerifiedChordShapes:
    def test_no_verified_data_passes_through(self):
        notes = [{"start": 1.0, "pitch": 45, "string": 1, "fret": 0}]
        result = _apply_verified_chord_shapes(notes, [(0.0, 5.0, "Am")], {})
        assert result == notes

    def test_overrides_matching_pitch_under_matching_chord(self):
        # Algorithm put A2 (pitch 45) on low E fret 5. Verified Am voicing
        # says it should be on A open (string 1, fret 0).
        notes = [{"start": 1.0, "pitch": 45, "string": 0, "fret": 5}]
        verified = {
            "verified": {
                "Am": {
                    "voicing": [{"midi_pitch": 45, "string": 1, "fret": 0}],
                    "applies_to": "all_spans",
                }
            }
        }
        chord_spans = [(0.0, 5.0, "Am")]
        result = _apply_verified_chord_shapes(notes, chord_spans, verified)
        assert result[0]["string"] == 1
        assert result[0]["fret"] == 0
        assert result[0]["overridden_by"] == "verified-shape:Am"

    def test_passes_through_when_chord_doesnt_match(self):
        # The chord active at this time is G, not Am — verification for Am
        # doesn't apply.
        notes = [{"start": 1.0, "pitch": 45, "string": 0, "fret": 5}]
        verified = {
            "verified": {
                "Am": {
                    "voicing": [{"midi_pitch": 45, "string": 1, "fret": 0}],
                    "applies_to": "all_spans",
                }
            }
        }
        chord_spans = [(0.0, 5.0, "G")]
        result = _apply_verified_chord_shapes(notes, chord_spans, verified)
        # No override — algorithm's choice stays.
        assert result[0]["string"] == 0
        assert result[0]["fret"] == 5

    def test_passes_through_when_pitch_not_in_voicing(self):
        notes = [{"start": 1.0, "pitch": 60, "string": 4, "fret": 1}]
        verified = {
            "verified": {
                "Am": {
                    # Voicing has pitch 45 but not 60.
                    "voicing": [{"midi_pitch": 45, "string": 1, "fret": 0}],
                    "applies_to": "all_spans",
                }
            }
        }
        result = _apply_verified_chord_shapes(notes, [(0.0, 5.0, "Am")], verified)
        assert result[0]["fret"] == 1

    def test_applies_to_specific_time_range(self):
        # Voicing should only apply during 0-2s.
        notes = [
            {"start": 1.0, "pitch": 45, "string": 0, "fret": 5},
            {"start": 3.0, "pitch": 45, "string": 0, "fret": 5},
        ]
        verified = {
            "verified": {
                "Am": {
                    "voicing": [{"midi_pitch": 45, "string": 1, "fret": 0}],
                    "applies_to": [{"start": 0.0, "end": 2.0}],
                }
            }
        }
        chord_spans = [(0.0, 5.0, "Am")]
        result = _apply_verified_chord_shapes(notes, chord_spans, verified)
        # First note: in range, overridden.
        assert result[0]["string"] == 1
        # Second note: outside range, untouched.
        assert result[1]["string"] == 0

    def test_skips_narrative_entries_without_voicing(self):
        # The narrative format (no structured voicing) should be skipped.
        notes = [{"start": 1.0, "pitch": 45, "string": 0, "fret": 5}]
        verified = {
            "verified": {
                "Am": {
                    "voicing_description": "cowboy Am",  # narrative
                    "matches_default_template": True,
                }
            }
        }
        result = _apply_verified_chord_shapes(notes, [(0.0, 5.0, "Am")], verified)
        # No override applied.
        assert result[0]["fret"] == 5
