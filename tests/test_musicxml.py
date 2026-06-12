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


class TestMusicxmlArticulations:
    """Articulation notations: slur + hammer-on/pull-off pairs, slides,
    bends, natural harmonics — and byte-stability without articulations."""

    def _make_inputs(self):
        sections_data = {"video_id": "test123"}
        rendered = [
            RenderedSection(
                label="riff",
                description="",
                canonical_start=0.0,
                canonical_end=8.0,
                chord_progression=["A"],
                cluster_count=4,
                note_count=4,
                ascii_tab="(unused)",
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
            "riff": [
                {"start": 0.0, "pitch": 45, "string": 1, "fret": 0, "note_index": 0},
                {"start": 1.0, "pitch": 48, "string": 1, "fret": 3, "note_index": 1},
                {"start": 2.0, "pitch": 58, "string": 3, "fret": 3, "note_index": 2},
                {"start": 3.0, "pitch": 57, "string": 1, "fret": 12, "note_index": 3},
            ]
        }
        beat_times_by_section = {"riff": [float(t) for t in range(9)]}
        return sections_data, rendered, tuning, notes_by_section, beat_times_by_section

    def test_hammer_emits_slur_and_technical_pair(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        artics = [
            {
                "type": "hammer",
                "from_note_index": 0,
                "note_index": 1,
                "string": 1,
                "from_fret": 0,
                "to_fret": 3,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        slurs = root.findall(".//slur")
        assert [s.get("type") for s in slurs] == ["start", "stop"]
        hammers = root.findall(".//technical/hammer-on")
        assert [h.get("type") for h in hammers] == ["start", "stop"]
        assert hammers[0].text == "H"
        assert hammers[1].text is None

    def test_pull_emits_pull_off_elements(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        # A real pull-off's source precedes its destination in time:
        # e|3 (idx 10, t=4) → e|0 (idx 11, t=5).
        ns["riff"].extend(
            [
                {"start": 4.0, "pitch": 67, "string": 5, "fret": 3, "note_index": 10},
                {"start": 5.0, "pitch": 64, "string": 5, "fret": 0, "note_index": 11},
            ]
        )
        artics = [
            {
                "type": "pull",
                "from_note_index": 10,
                "note_index": 11,
                "string": 5,
                "from_fret": 3,
                "to_fret": 0,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        pulls = root.findall(".//technical/pull-off")
        assert [p.get("type") for p in pulls] == ["start", "stop"]
        assert pulls[0].text == "P"

    def test_slide_emits_slide_elements(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        artics = [
            {
                "type": "slide",
                "from_note_index": 0,
                "note_index": 1,
                "string": 1,
                "from_fret": 0,
                "to_fret": 3,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        slides = root.findall(".//slide")
        assert [s.get("type") for s in slides] == ["start", "stop"]

    def test_bend_and_harmonic_marks(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        artics = [
            {
                "type": "bend",
                "note_index": 2,
                "string": 3,
                "fret": 3,
                "target_semitones": 2,
                "member_note_indices": [],
                "evidence": {},
            },
            {"type": "harmonic", "note_index": 3, "open_string": 1, "node_fret": 12},
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        alter = root.find(".//technical/bend/bend-alter")
        assert alter is not None
        assert alter.text == "2"
        assert root.find(".//technical/harmonic/natural") is not None

    def test_restrung_pair_half_emits_no_marks(self):
        # The TO note was moved to another string (what a verified-shape
        # override does). A cross-string hammer-on is impossible notation
        # and the same run's tab.txt drops the connector — the XML must
        # agree instead of emitting slur + hammer-on anyway.
        sd, rd, tu, ns, bs = self._make_inputs()
        ns["riff"][1]["string"] = 2
        artics = [
            {
                "type": "hammer",
                "from_note_index": 0,
                "note_index": 1,
                "string": 1,
                "from_fret": 0,
                "to_fret": 3,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.findall(".//slur") == []
        assert root.findall(".//technical/hammer-on") == []

    def test_refretted_pair_half_emits_no_marks(self):
        # Same gating on the fret axis: the TO note's emitted fret no
        # longer matches the articulation's to_fret evidence.
        sd, rd, tu, ns, bs = self._make_inputs()
        ns["riff"][1]["fret"] = 5
        artics = [
            {
                "type": "hammer",
                "from_note_index": 0,
                "note_index": 1,
                "string": 1,
                "from_fret": 0,
                "to_fret": 3,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.findall(".//slur") == []
        assert root.findall(".//technical/hammer-on") == []

    def test_bend_on_moved_note_emits_no_bend(self):
        # The struck note no longer sits at the measured (string, fret) —
        # the bend evidence is void, mirroring the ASCII renderer's
        # "bend mark dropped" behavior.
        sd, rd, tu, ns, bs = self._make_inputs()
        ns["riff"][2]["string"] = 4
        ns["riff"][2]["fret"] = 8
        artics = [
            {
                "type": "bend",
                "note_index": 2,
                "string": 3,
                "fret": 3,
                "target_semitones": 2,
                "member_note_indices": [],
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.findall(".//technical/bend") == []

    def test_harmonic_not_at_node_position_emits_no_harmonic(self):
        # The prelayout pass re-strings a valid harmonic to its
        # (open_string, node_fret) before notes reach the exporter; a
        # note still at the Viterbi position means the entry was rejected
        # there, so no <natural/> here either.
        sd, rd, tu, ns, bs = self._make_inputs()
        ns["riff"][3]["string"] = 3
        ns["riff"][3]["fret"] = 2
        artics = [{"type": "harmonic", "note_index": 3, "open_string": 1, "node_fret": 12}]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.find(".//technical/harmonic") is None

    def test_orphaned_pair_emits_no_dangling_slur(self):
        # One half references a note that is not being emitted (e.g. a
        # hidden bend member or a filtered note) — no slur at all.
        sd, rd, tu, ns, bs = self._make_inputs()
        artics = [
            {
                "type": "hammer",
                "from_note_index": 0,
                "note_index": 99,
                "string": 1,
                "from_fret": 0,
                "to_fret": 3,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.findall(".//slur") == []
        assert root.findall(".//technical/hammer-on") == []

    def test_malformed_entries_never_crash(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        artics = [{"type": "bend"}, {"type": "hammer", "note_index": "x"}, {"type": "?"}]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.tag == "score-partwise"

    def test_no_articulations_byte_identical(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        base = render_musicxml(sd, rd, tu, ns, bs)
        assert render_musicxml(sd, rd, tu, ns, bs, articulations=None) == base
        assert render_musicxml(sd, rd, tu, ns, bs, articulations=[]) == base

    def test_notes_without_note_index_tolerated(self):
        sd, rd, tu, ns, bs = self._make_inputs()
        for n in ns["riff"]:
            del n["note_index"]
        artics = [
            {
                "type": "hammer",
                "from_note_index": 0,
                "note_index": 1,
                "string": 1,
                "from_fret": 0,
                "to_fret": 3,
                "evidence": {},
            }
        ]
        root = ET.fromstring(render_musicxml(sd, rd, tu, ns, bs, articulations=artics))
        assert root.findall(".//slur") == []
