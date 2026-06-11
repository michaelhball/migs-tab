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


def _note(start, end, pitch, string, fret, cluster_id):
    return {
        "note_index": 0,
        "start": start,
        "end": end,
        "pitch": pitch,
        "string": string,
        "fret": fret,
        "cluster_id": cluster_id,
        "ambiguous": False,
    }


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
