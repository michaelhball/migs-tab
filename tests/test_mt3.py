"""Tests for the YourMT3+ wrapper — guitar-program filtering + JSON shape."""

from __future__ import annotations

import json

import pretty_midi
import pytest

from migs_tab.mt3 import (
    _GUITAR_PROGRAMS,
    _MAX_GUITAR_MIDI,
    _MIN_GUITAR_MIDI,
    _is_voice_channel,
    _resolve_audio_source,
    _write_notes_json,
)
from migs_tab.paths import VideoPaths


def _make_midi(path, instruments: list[tuple[int, str, bool, list[tuple[float, float, int]]]]):
    """Build a multi-instrument MIDI fixture.

    Each tuple: (program, name, is_drum, [(start, end, pitch)...]).
    """
    pm = pretty_midi.PrettyMIDI()
    for program, name, is_drum, notes in instruments:
        inst = pretty_midi.Instrument(program=program, name=name, is_drum=is_drum)
        for start, end, pitch in notes:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
        pm.instruments.append(inst)
    pm.write(str(path))


class TestGuitarProgramConstant:
    def test_covers_all_gm_guitar_programs(self):
        assert _GUITAR_PROGRAMS == frozenset(range(24, 32))


class TestWriteNotesJson:
    def test_keeps_all_non_drum_channels(self, tmp_path):
        midi = tmp_path / "x.mid"
        _make_midi(
            midi,
            [
                (0, "Acoustic Piano", False, [(0.0, 0.5, 60)]),  # kept
                (24, "Nylon Guitar", False, [(0.1, 0.6, 64)]),  # kept
                (27, "Clean Guitar", False, [(0.2, 0.7, 67)]),  # kept
                (33, "Bass", False, [(0.3, 0.8, 36)]),  # kept (real low note)
                (0, "Drums", True, [(0.0, 0.1, 38)]),  # dropped (is_drum)
            ],
        )
        out_json = tmp_path / "x.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        pitches = sorted(n["pitch"] for n in data["notes"])
        assert pitches == [36, 60, 64, 67]
        assert data["backend"] == "mt3"
        assert "dedupe" in data["filter"]

    def test_dedupes_cross_channel_duplicates(self, tmp_path):
        """Same pitch fired by Guitar and Piano within tolerance = one note."""
        midi = tmp_path / "dup.mid"
        _make_midi(
            midi,
            [
                (27, "Clean Guitar", False, [(0.50, 1.0, 60)]),
                (0, "Acoustic Piano", False, [(0.52, 1.0, 60)]),  # within ±50ms
                (2, "Electric Piano", False, [(0.51, 1.0, 60)]),  # within ±50ms
            ],
        )
        out_json = tmp_path / "dup.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        # All three collapse into the earliest-start single note at pitch 60.
        starts_pitches = [(n["start"], n["pitch"]) for n in data["notes"]]
        assert starts_pitches == [(0.5, 60)]

    def test_keeps_distinct_attacks_outside_tolerance(self, tmp_path):
        """Two attacks at the same pitch far apart = two notes."""
        midi = tmp_path / "two.mid"
        _make_midi(
            midi,
            [
                (27, "Clean Guitar", False, [(0.50, 0.9, 60), (2.00, 2.5, 60)]),
            ],
        )
        out_json = tmp_path / "two.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        starts = [n["start"] for n in data["notes"]]
        assert starts == [0.5, 2.0]

    def test_records_full_breakdown(self, tmp_path):
        midi = tmp_path / "y.mid"
        _make_midi(
            midi,
            [
                (0, "Piano", False, [(0.0, 0.5, 60), (0.5, 1.0, 62)]),
                (27, "Clean Guitar", False, [(0.1, 0.6, 64)]),
            ],
        )
        out_json = tmp_path / "y.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        breakdown = {b["name"]: b["note_count"] for b in data["instrument_breakdown"]}
        assert breakdown["Piano"] == 2
        assert breakdown["Clean Guitar"] == 1

    def test_sorted_by_start_then_pitch(self, tmp_path):
        midi = tmp_path / "z.mid"
        _make_midi(
            midi,
            [
                (
                    27,
                    "Clean",
                    False,
                    [(1.0, 1.5, 60), (0.5, 1.0, 64), (0.5, 1.0, 60)],
                )
            ],
        )
        out_json = tmp_path / "z.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        starts_pitches = [(n["start"], n["pitch"]) for n in data["notes"]]
        # (0.5, 60) before (0.5, 64) before (1.0, 60)
        assert starts_pitches == [(0.5, 60), (0.5, 64), (1.0, 60)]

    def test_empty_when_only_drums(self, tmp_path):
        """Drum-only MIDI yields no melodic notes — GM drum pitches are kit codes, not music."""
        midi = tmp_path / "drums.mid"
        _make_midi(midi, [(0, "Drums", True, [(0.0, 0.1, 38), (0.5, 0.6, 42)])])
        out_json = tmp_path / "drums.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        assert data["notes"] == []
        assert data["note_count"] == 0


class TestVoiceFilter:
    def test_drops_singing_voice_channel(self, tmp_path):
        """MT3 writes instructor speech as program 65 / 'Singing Voice' — drop it whole."""
        midi = tmp_path / "voice.mid"
        _make_midi(
            midi,
            [
                (24, "Guitar (clean)", False, [(0.0, 0.5, 52), (1.0, 1.5, 57)]),
                (65, "Singing Voice", False, [(0.0, 0.3, 60), (0.4, 0.7, 62), (0.8, 1.1, 64)]),
            ],
        )
        out_json = tmp_path / "voice.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        pitches = sorted(n["pitch"] for n in data["notes"])
        assert pitches == [52, 57]
        assert data["filtered_voice_notes"] == 3
        assert data["filtered_voice_channels"] == [
            {"program": 65, "name": "Singing Voice", "note_count": 3}
        ]

    def test_drops_singing_voice_chorus_channel(self, tmp_path):
        """The chorus variant maps to GM program 53 with a 'Singing Voice (chorus)' name."""
        midi = tmp_path / "chorus.mid"
        _make_midi(
            midi,
            [
                (27, "Guitar (clean)", False, [(0.0, 0.5, 52)]),
                (53, "Singing Voice (chorus)", False, [(0.0, 0.3, 60)]),
            ],
        )
        out_json = tmp_path / "chorus.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        assert [n["pitch"] for n in data["notes"]] == [52]
        assert data["filtered_voice_notes"] == 1

    def test_keeps_mislabeled_melodic_channels(self, tmp_path):
        """Piano/Strings channels on a guitar stem are MISLABELED REAL GUITAR — keep them."""
        midi = tmp_path / "mislabeled.mid"
        _make_midi(
            midi,
            [
                (0, "Acoustic Piano", False, [(0.0, 0.5, 60)]),
                (2, "Electric Piano", False, [(1.0, 1.5, 62)]),
                (48, "Strings", False, [(2.0, 2.5, 64)]),
                (65, "Singing Voice", False, [(3.0, 3.3, 66)]),
            ],
        )
        out_json = tmp_path / "mislabeled.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        pitches = sorted(n["pitch"] for n in data["notes"])
        assert pitches == [60, 62, 64]
        assert data["filtered_voice_notes"] == 1

    def test_name_alone_marks_voice_channel(self, tmp_path):
        """The track name is authoritative — voice is dropped even with program 0."""
        midi = tmp_path / "name_only.mid"
        _make_midi(
            midi,
            [
                (24, "Guitar (clean)", False, [(0.0, 0.5, 52)]),
                (0, "Singing Voice", False, [(0.0, 0.3, 60)]),
            ],
        )
        out_json = tmp_path / "name_only.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        assert [n["pitch"] for n in data["notes"]] == [52]
        assert data["filtered_voice_notes"] == 1

    def test_voice_programs_with_real_instrument_names_kept(self, tmp_path):
        """full_plus vocabs legitimately emit 'Voice Oohs' [53] and 'Alto Sax' [65].

        Those channels are usually mislabeled real guitar — only a 'Singing
        Voice' track name may drop a channel when a name is present.
        """
        midi = tmp_path / "full_plus.mid"
        _make_midi(
            midi,
            [
                (24, "Guitar (clean)", False, [(0.0, 0.5, 52)]),
                (65, "Alto Sax", False, [(1.0, 1.5, 60)]),
                (53, "Voice Oohs", False, [(2.0, 2.5, 62)]),
            ],
        )
        out_json = tmp_path / "full_plus.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        pitches = sorted(n["pitch"] for n in data["notes"])
        assert pitches == [52, 60, 62]
        assert data["filtered_voice_notes"] == 0
        assert data["filtered_voice_channels"] == []

    def test_program_fallback_only_when_name_missing(self):
        """The program set is consulted solely for nameless tracks."""
        assert _is_voice_channel("Singing Voice", 0) is True
        assert _is_voice_channel("Singing Voice (chorus)", 53) is True
        assert _is_voice_channel("Alto Sax", 65) is False
        assert _is_voice_channel("Voice Oohs", 53) is False
        assert _is_voice_channel("", 65) is True
        assert _is_voice_channel("", 53) is True
        assert _is_voice_channel("", 24) is False

    def test_voice_channel_still_listed_in_breakdown(self, tmp_path):
        """Filtering must not erase the channel from instrument_breakdown."""
        midi = tmp_path / "audit.mid"
        _make_midi(
            midi,
            [
                (24, "Guitar (clean)", False, [(0.0, 0.5, 52)]),
                (65, "Singing Voice", False, [(0.0, 0.3, 60)]),
            ],
        )
        out_json = tmp_path / "audit.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        breakdown = {b["name"]: b["note_count"] for b in data["instrument_breakdown"]}
        assert breakdown["Singing Voice"] == 1


class TestPitchClamp:
    def test_constants(self):
        # 36 = C2 (Drop D half-step down is C#2=37); 88 = E6 (24th fret high E).
        assert _MIN_GUITAR_MIDI == 36
        assert _MAX_GUITAR_MIDI == 88

    def test_boundaries_inclusive(self, tmp_path):
        """Pitches exactly at 36 and 88 are playable and must be kept."""
        midi = tmp_path / "bounds.mid"
        _make_midi(
            midi,
            [
                (
                    24,
                    "Guitar (clean)",
                    False,
                    [
                        (0.0, 0.5, _MIN_GUITAR_MIDI - 1),  # 35 — dropped
                        (1.0, 1.5, _MIN_GUITAR_MIDI),  # 36 — kept
                        (2.0, 2.5, _MAX_GUITAR_MIDI),  # 88 — kept
                        (3.0, 3.5, _MAX_GUITAR_MIDI + 1),  # 89 — dropped
                    ],
                )
            ],
        )
        out_json = tmp_path / "bounds.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        pitches = sorted(n["pitch"] for n in data["notes"])
        assert pitches == [_MIN_GUITAR_MIDI, _MAX_GUITAR_MIDI]
        assert data["filtered_out_of_range_notes"] == 2
        assert data["pitch_range"] == [_MIN_GUITAR_MIDI, _MAX_GUITAR_MIDI]

    def test_extreme_artifacts_dropped(self, tmp_path):
        """Cached MT3 output spans MIDI 26-100; the extremes are artifacts."""
        midi = tmp_path / "extreme.mid"
        _make_midi(
            midi,
            [
                (24, "Guitar (clean)", False, [(0.0, 0.5, 26), (1.0, 1.5, 100), (2.0, 2.5, 52)]),
            ],
        )
        out_json = tmp_path / "extreme.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        assert [n["pitch"] for n in data["notes"]] == [52]
        assert data["filtered_out_of_range_notes"] == 2


class TestProvenance:
    def test_records_audio_source_relative_to_cache_root(self, tmp_path):
        cache_root = tmp_path / "cache" / "vid123"
        stem = cache_root / "stems" / "other.wav"
        stem.parent.mkdir(parents=True)
        stem.write_bytes(b"\x00" * 128)

        midi = tmp_path / "p.mid"
        _make_midi(midi, [(24, "Guitar (clean)", False, [(0.0, 0.5, 52)])])
        out_json = tmp_path / "p.json"
        _write_notes_json(
            midi,
            out_json,
            audio_source=stem,
            cache_root=cache_root,
            variant="YMT3+",
            batch_size=2,
        )

        data = json.loads(out_json.read_text())
        prov = data["provenance"]
        assert prov["audio_source"] == "stems/other.wav"
        assert prov["audio_size_bytes"] == 128
        assert prov["audio_mtime"]  # ISO timestamp, non-empty
        assert prov["model_variant"] == "YMT3+"
        assert prov["batch_size"] == 2

    def test_absolute_path_kept_when_outside_cache_root(self, tmp_path):
        elsewhere = tmp_path / "elsewhere.wav"
        elsewhere.write_bytes(b"\x00" * 16)
        cache_root = tmp_path / "cache" / "vid123"
        cache_root.mkdir(parents=True)

        midi = tmp_path / "q.mid"
        _make_midi(midi, [(24, "Guitar (clean)", False, [(0.0, 0.5, 52)])])
        out_json = tmp_path / "q.json"
        _write_notes_json(midi, out_json, audio_source=elsewhere, cache_root=cache_root)

        data = json.loads(out_json.read_text())
        assert data["provenance"]["audio_source"] == str(elsewhere)

    def test_provenance_present_with_nulls_when_unknown(self, tmp_path):
        """Legacy call path without source info still writes the provenance block."""
        midi = tmp_path / "r.mid"
        _make_midi(midi, [(24, "Guitar (clean)", False, [(0.0, 0.5, 52)])])
        out_json = tmp_path / "r.json"
        _write_notes_json(midi, out_json)

        data = json.loads(out_json.read_text())
        prov = data["provenance"]
        assert prov["audio_source"] is None
        assert prov["audio_mtime"] is None
        assert prov["audio_size_bytes"] is None
        assert prov["model_variant"] is None
        assert prov["batch_size"] is None


class TestResolveAudioSource:
    """Priority: explicit override > Demucs guitar stem > raw mix fallback."""

    def _paths(self, tmp_path) -> VideoPaths:
        return VideoPaths("testvideo01", cache_dir=tmp_path)

    def test_explicit_override_wins(self, tmp_path):
        paths = self._paths(tmp_path)
        paths.stems_dir.mkdir(parents=True)
        paths.guitar_stem.write_bytes(b"\x00")
        paths.audio.write_bytes(b"\x00")
        override = tmp_path / "clip.wav"
        override.write_bytes(b"\x00")

        assert _resolve_audio_source(paths, override) == override

    def test_prefers_guitar_stem_without_override(self, tmp_path):
        paths = self._paths(tmp_path)
        paths.stems_dir.mkdir(parents=True)
        paths.guitar_stem.write_bytes(b"\x00")
        paths.audio.write_bytes(b"\x00")

        assert _resolve_audio_source(paths, None) == paths.guitar_stem

    def test_falls_back_to_raw_mix_when_stem_missing(self, tmp_path):
        paths = self._paths(tmp_path)
        paths.audio.write_bytes(b"\x00")

        assert _resolve_audio_source(paths, None) == paths.audio

    def test_raises_when_no_audio_exists(self, tmp_path):
        paths = self._paths(tmp_path)

        with pytest.raises(FileNotFoundError):
            _resolve_audio_source(paths, None)

    def test_raises_when_override_missing_despite_fallbacks(self, tmp_path):
        """An explicit override is authoritative — no silent fallback to the mix."""
        paths = self._paths(tmp_path)
        paths.audio.write_bytes(b"\x00")

        with pytest.raises(FileNotFoundError):
            _resolve_audio_source(paths, tmp_path / "missing.wav")
