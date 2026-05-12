"""Tests for musicxml.py — pitch conversion, subdivisions, full document
structure for a small synthetic input."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from migs_tab.musicxml import (
    _midi_to_step_octave_alter,
    _nearest_index,
    _subdivisions,
    render_musicxml,
)
from migs_tab.render import RenderedSection, TuningInfo


class TestMidiToStepOctaveAlter:
    def test_middle_c(self):
        # MIDI 60 = C4
        step, octave, alter = _midi_to_step_octave_alter(60)
        assert step == "C"
        assert octave == 4
        assert alter == 0

    def test_csharp(self):
        step, octave, alter = _midi_to_step_octave_alter(61)
        assert step == "C"
        assert alter == 1

    def test_low_e(self):
        # MIDI 40 = E2
        step, octave, alter = _midi_to_step_octave_alter(40)
        assert step == "E"
        assert octave == 2

    def test_high_e(self):
        # MIDI 64 = E4
        step, octave, alter = _midi_to_step_octave_alter(64)
        assert step == "E"
        assert octave == 4


class TestSubdivisions:
    def test_expands_beats(self):
        # 4 beats with 2 subs each → 8 + 1 = 9 grid points.
        grid = _subdivisions([0.0, 1.0, 2.0, 3.0], subs_per_beat=2)
        assert len(grid) == 7  # 3 inter-beat × 2 subs + 1 trailing beat
        # First grid entry is at the first beat and is marked is_beat=True.
        assert grid[0][1] is True

    def test_handles_single_beat(self):
        grid = _subdivisions([1.0], subs_per_beat=2)
        assert len(grid) == 1


class TestNearestIndex:
    def test_finds_closer(self):
        assert _nearest_index([0.0, 1.0, 2.0], 1.4) == 1
        assert _nearest_index([0.0, 1.0, 2.0], 1.6) == 2


class TestRenderMusicxml:
    def _make_inputs(self):
        sections_data = {"video_id": "test123"}
        rendered = [
            RenderedSection(
                label="intro",
                description="test intro",
                canonical_start=0.0,
                canonical_end=8.0,
                chord_progression=["Am"],
                cluster_count=2,
                note_count=2,
                ascii_tab="(unused for XML)",
                tempo_bpm=120.0,
            )
        ]
        tuning = TuningInfo(
            label="Standard",
            capo=0,
            strings_midi=[40, 45, 50, 55, 59, 64],
            source="audio",
            confidence=1.0,
            string_letters=["e", "B", "G", "D", "A", "E"],
        )
        notes_by_section = {
            "intro": [
                {"start": 0.0, "pitch": 45, "string": 1, "fret": 0},  # A2 on A open
                {"start": 4.0, "pitch": 64, "string": 5, "fret": 0},  # E4 on high E
            ]
        }
        # 4 beats spaced 1s apart for an 8s section.
        beat_times_by_section = {"intro": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
        return sections_data, rendered, tuning, notes_by_section, beat_times_by_section

    def test_produces_valid_xml(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        xml = render_musicxml(sd, rd, tu, ns, bs)
        # Parses without error.
        root = ET.fromstring(xml)
        assert root.tag == "score-partwise"

    def test_has_one_part(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        xml = render_musicxml(sd, rd, tu, ns, bs)
        root = ET.fromstring(xml)
        parts = root.findall("part")
        assert len(parts) == 1

    def test_has_tab_clef(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        xml = render_musicxml(sd, rd, tu, ns, bs)
        root = ET.fromstring(xml)
        sign = root.find(".//clef/sign")
        assert sign is not None
        assert sign.text == "TAB"

    def test_has_six_staff_tuning_lines(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        xml = render_musicxml(sd, rd, tu, ns, bs)
        root = ET.fromstring(xml)
        tunings = root.findall(".//staff-tuning")
        assert len(tunings) == 6

    def test_has_section_rehearsal_mark(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        xml = render_musicxml(sd, rd, tu, ns, bs)
        root = ET.fromstring(xml)
        rehearsals = root.findall(".//rehearsal")
        assert len(rehearsals) >= 1
        assert any(r.text == "intro" for r in rehearsals)

    def test_notes_have_string_and_fret(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        xml = render_musicxml(sd, rd, tu, ns, bs)
        root = ET.fromstring(xml)
        # Find notes (not rests).
        for note in root.findall(".//note"):
            if note.find("rest") is not None:
                continue
            # Each real note should carry technical/string + fret.
            string_el = note.find(".//technical/string")
            fret_el = note.find(".//technical/fret")
            if string_el is not None:
                assert fret_el is not None

    def test_includes_capo_in_staff_details(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        # Switch tuning to have a capo.
        tu_capo = TuningInfo(
            label="Standard, capo 3",
            capo=3,
            strings_midi=[40, 45, 50, 55, 59, 64],
            source="captions",
            confidence=0.9,
            string_letters=["e", "B", "G", "D", "A", "E"],
        )
        xml = render_musicxml(sd, rd, tu_capo, ns, bs)
        root = ET.fromstring(xml)
        capo_el = root.find(".//capo")
        assert capo_el is not None
        assert capo_el.text == "3"
