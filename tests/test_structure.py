"""Tests for structure.py — VTT parsing, chord templates, structure data."""

from __future__ import annotations

import numpy as np

from migs_tab.structure import (
    PITCH_NAMES,
    Caption,
    _build_chord_templates,
    _captions_for_segment,
    _parse_vtt,
)

VTT_SAMPLE = """WEBVTT
Kind: captions
Language: en

00:00:01.500 --> 00:00:04.000 align:start position:0%
okay so this is

00:00:04.000 --> 00:00:06.500 align:start position:0%
the first chord <c.color>Am</c>

NOTE this is an editorial note

00:00:07.000 --> 00:00:09.500 align:start position:0%
the first chord Am
"""


class TestParseVtt:
    def test_strips_tags(self, tmp_path):
        path = tmp_path / "captions.vtt"
        path.write_text(VTT_SAMPLE)
        captions = _parse_vtt(path)
        # All entries should have no embedded tags.
        for c in captions:
            assert "<c" not in c.text
            assert "</c>" not in c.text

    def test_skips_webvtt_header(self, tmp_path):
        path = tmp_path / "captions.vtt"
        path.write_text(VTT_SAMPLE)
        captions = _parse_vtt(path)
        for c in captions:
            assert "WEBVTT" not in c.text
            assert "Kind:" not in c.text

    def test_skips_note_lines(self, tmp_path):
        path = tmp_path / "captions.vtt"
        path.write_text(VTT_SAMPLE)
        captions = _parse_vtt(path)
        for c in captions:
            assert "this is an editorial note" not in c.text

    def test_dedupes_repeat_text(self, tmp_path):
        # The sample has "the first chord Am" twice — should dedupe.
        path = tmp_path / "captions.vtt"
        path.write_text(VTT_SAMPLE)
        captions = _parse_vtt(path)
        texts = [c.text for c in captions]
        assert texts.count("the first chord Am") <= 1

    def test_timestamps_parsed(self, tmp_path):
        path = tmp_path / "captions.vtt"
        path.write_text(VTT_SAMPLE)
        captions = _parse_vtt(path)
        # First caption starts at 1.5s.
        assert any(abs(c.start - 1.5) < 0.001 for c in captions)


class TestCaptionsForSegment:
    def test_overlapping_kept(self):
        caps = [
            Caption(start=10.0, end=15.0, text="first"),
            Caption(start=20.0, end=25.0, text="second"),
            Caption(start=30.0, end=35.0, text="third"),
        ]
        result = _captions_for_segment(caps, 18.0, 28.0)
        # "first" ends at 15 — within the 5s lookback window before 18, so it's included.
        assert any(c.text == "first" for c in result)
        # "second" overlaps the window.
        assert any(c.text == "second" for c in result)
        # "third" is fully after the window.
        assert not any(c.text == "third" for c in result)


class TestBuildChordTemplates:
    def test_36_chord_templates(self):
        # 12 roots × 3 qualities (maj, min, dom7) = 36 templates.
        templates = _build_chord_templates()
        assert len(templates) == 36

    def test_template_values_are_normalized(self):
        templates = _build_chord_templates()
        for name, vec in templates.items():
            assert vec.shape == (12,)
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-6, f"{name} should be unit-normalized"

    def test_major_includes_root_third_fifth(self):
        templates = _build_chord_templates()
        c_major = templates["C"]
        # Major triad: root, M3, P5 = C, E, G = pitch classes 0, 4, 7
        # Vector should have non-zero entries at those positions.
        assert c_major[0] > 0
        assert c_major[4] > 0
        assert c_major[7] > 0
        # Zero elsewhere.
        assert c_major[1] == 0
        assert c_major[3] == 0

    def test_minor_includes_minor_third(self):
        templates = _build_chord_templates()
        c_minor = templates["Cm"]
        assert c_minor[0] > 0
        assert c_minor[3] > 0  # m3
        assert c_minor[7] > 0  # P5
        assert c_minor[4] == 0  # no major 3rd

    def test_dom7_includes_minor_seventh(self):
        templates = _build_chord_templates()
        c_dom7 = templates["C7"]
        # C7 = C E G Bb = pitch classes 0, 4, 7, 10
        assert c_dom7[0] > 0
        assert c_dom7[4] > 0
        assert c_dom7[7] > 0
        assert c_dom7[10] > 0


def test_pitch_names_in_canonical_order():
    assert PITCH_NAMES == [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
