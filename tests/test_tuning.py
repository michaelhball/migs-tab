"""Tests for tuning.py — caption pattern matching, library defaults, audio
candidate generation, transcription contradiction veto."""

from __future__ import annotations

import json

from migs_tab.paths import VideoPaths
from migs_tab.tuning import (
    _AUDIO_CANDIDATES,
    _MIN_FLOOR_NOTES,
    _REFIT_MAX_SKIP_SEMITONES,
    _TUNING_LIBRARY,
    _UNVERIFIED_CONFIDENCE_CAP,
    _VETOED_CONFIDENCE_CAP,
    STANDARD_TUNING,
    Tuning,
    _detect_from_captions,
    default_tuning,
    detect_tuning,
    load_transcribed_notes,
    verify_against_transcription,
)


class TestTuningLibrary:
    def test_has_standard(self):
        assert _TUNING_LIBRARY["Standard"] == [40, 45, 50, 55, 59, 64]

    def test_has_drop_d(self):
        assert _TUNING_LIBRARY["Drop D"] == [38, 45, 50, 55, 59, 64]

    def test_has_double_drop_d(self):
        assert _TUNING_LIBRARY["Double Drop D"] == [38, 45, 50, 55, 59, 62]

    def test_has_dadgad(self):
        assert _TUNING_LIBRARY["DADGAD"] == [38, 45, 50, 55, 57, 62]

    def test_all_six_strings(self):
        for label, midis in _TUNING_LIBRARY.items():
            assert len(midis) == 6, f"{label} should have 6 strings"

    def test_low_to_high_ordering(self):
        for label, midis in _TUNING_LIBRARY.items():
            # Strings should be roughly ascending. We allow some exceptions
            # (like Open G's high D being lower than its B, etc.) but the
            # majority must be ascending.
            ascending = sum(1 for i in range(5) if midis[i + 1] >= midis[i])
            assert ascending >= 4, f"{label} should be mostly low→high"


class TestAudioCandidates:
    def test_includes_standard_capo_range(self):
        labels = [c[0] for c in _AUDIO_CANDIDATES]
        # Standard appears multiple times for capo 0..7.
        std_count = sum(1 for lbl in labels if lbl == "Standard")
        assert std_count >= 5  # at least capos 0..4

    def test_includes_alternate_tunings(self):
        labels = [c[0] for c in _AUDIO_CANDIDATES]
        for required in ("Drop D", "Double Drop D", "DADGAD", "Open D", "Open G", "Open E"):
            assert required in labels


class TestDetectFromCaptions:
    def test_drop_d_with_tuning_context(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("Okay, today we're in drop D and we'll play...")
        result = _detect_from_captions(paths)
        assert result is not None
        assert result.label == "Drop D"
        assert result.strings_midi == _TUNING_LIBRARY["Drop D"]

    def test_double_drop_d_wins_over_drop_d(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("The song is in double drop d tuning")
        result = _detect_from_captions(paths)
        assert result is not None
        assert result.label == "Double Drop D"

    def test_dadgad_unambiguous(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("This is in DADGAD")
        result = _detect_from_captions(paths)
        assert result is not None
        assert result.label == "DADGAD"

    def test_open_d_only_with_tuning_suffix(self, tmp_path):
        # "Open D" without "tuning" context should NOT match (could be talking
        # about open D string).
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("Play the open D string here")
        result = _detect_from_captions(paths)
        # Should NOT match Open D tuning from this.
        if result is not None:
            assert result.label != "Open D"

    def test_open_d_tuning_matches(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("This song is in Open D tuning")
        result = _detect_from_captions(paths)
        assert result is not None
        assert result.label == "Open D"

    def test_capo_alone_matches_standard_with_capo(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("Put your capo on the 3rd fret")
        result = _detect_from_captions(paths)
        assert result is not None
        assert result.capo == 3
        assert result.strings_midi == _TUNING_LIBRARY["Standard"]

    def test_half_step_down(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("Tune everything half-step down for this song")
        result = _detect_from_captions(paths)
        assert result is not None
        assert result.label == "Half-step down"

    def test_no_tuning_keyword_returns_none(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text(
            "We're going to play an Am chord, then an E7, then a G sus, then resolve back to C."
        )
        result = _detect_from_captions(paths)
        assert result is None

    def test_open_g_string_does_not_fire(self, tmp_path):
        # "open G string" is common parlance for the open G string, not the tuning.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("Strum the open G string and then move down")
        result = _detect_from_captions(paths)
        if result is not None:
            assert result.label != "Open G"

    def test_empty_captions_returns_none(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("")
        result = _detect_from_captions(paths)
        assert result is None


class TestDefaultTuning:
    def test_returns_standard(self):
        t = default_tuning("test")
        assert t.label == "Standard"
        assert t.strings_midi == list(STANDARD_TUNING)
        assert t.capo == 0


class TestTuningSerialization:
    def test_to_dict_roundtrip(self):
        t = Tuning(
            strings_midi=[40, 45, 50, 55, 59, 64],
            capo=2,
            label="Test",
            confidence=0.95,
            source="captions",
            evidence="...",
        )
        d = t.to_dict()
        assert d["strings_midi"] == [40, 45, 50, 55, 59, 64]
        assert d["capo"] == 2
        assert d["label"] == "Test"
        assert d["confidence"] == 0.95

    def test_effective_open_pitches_applies_capo(self):
        t = Tuning(
            strings_midi=[40, 45, 50, 55, 59, 64],
            capo=3,
            label="Standard, capo 3",
            confidence=1.0,
            source="captions",
            evidence="",
        )
        assert t.effective_open_pitches() == [43, 48, 53, 58, 62, 67]


def test_detect_tuning_writes_json_when_only_default_available(tmp_path):
    """Smoke test — when there's nothing to read from, detect_tuning writes
    a default-tuning JSON."""
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    # No captions, no audio — should fall through to default_tuning.
    detect_tuning(paths, force=True)
    assert paths.tuning_json.exists()
    data = json.loads(paths.tuning_json.read_text())
    assert data["label"] == "Standard"
    # With no transcription on disk the result must be flagged unverified.
    assert data["verification"]["checked"] is False


# ---------------------------------------------------------------------------
# Transcription contradiction veto
# ---------------------------------------------------------------------------


def _note(pitch: int, start: float, dur: float = 0.5, **extra) -> dict:
    n = {"start": start, "end": round(start + dur, 4), "pitch": pitch, "velocity": 100}
    n.update(extra)
    return n


def _write_mt3_notes(paths: VideoPaths, notes: list[dict]) -> None:
    paths.notes_mt3_json.write_text(
        json.dumps({"backend": "mt3", "note_count": len(notes), "notes": notes})
    )


def _write_basic_pitch_notes(paths: VideoPaths, notes: list[dict]) -> None:
    """Write notes to notes.json only (basic-pitch fallback source) so the
    cross-check loads them with notes_source == 'basic_pitch'."""
    paths.notes_json.write_text(
        json.dumps({"backend": "basic_pitch", "note_count": len(notes), "notes": notes})
    )


def _standard_capo5(confidence: float = 1.0) -> Tuning:
    return Tuning(
        strings_midi=list(STANDARD_TUNING),
        capo=5,
        label="Standard, capo 5",
        confidence=confidence,
        source="audio",
        evidence="lowest sustained pitch ≈ MIDI 45.00",
    )


def _body_notes(n: int = 100, lo: int = 45, hi: int = 70) -> list[dict]:
    """n unremarkable notes at/above the capo-5 floor."""
    return [_note(lo + (i % (hi - lo)), 10.0 + i * 0.5) for i in range(n)]


class TestContradictionVeto:
    def test_e2_rich_notes_veto_capo5_to_capo0(self, tmp_path):
        # Mirrors 9jswOBilMvA: dozens of sustained E2/F#2 notes below the
        # capo-5 floor (45) must flip the result to Standard, capo 0.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(100)
        notes += [_note(40, 100.0 + i, dur=0.6) for i in range(20)]  # E2
        notes += [_note(42, 130.0 + i, dur=0.5) for i in range(15)]  # F#2
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.capo == 0
        assert result.strings_midi == list(STANDARD_TUNING)
        assert result.label == "Standard"
        assert result.confidence <= _VETOED_CONFIDENCE_CAP
        assert "contradiction-veto" in result.source
        assert result.verification["veto"] is True
        assert result.verification["vetoed_label"] == "Standard, capo 5"
        assert result.verification["supported_floor_midi"] == 40
        assert result.verification["subfloor_sustained_count"] == 35

    def test_genuinely_capoed_input_keeps_capo(self, tmp_path):
        # All notes at/above the capo-5 floor: the capo is confirmed, the
        # confidence untouched, and the check is recorded in verification.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        _write_mt3_notes(paths, _body_notes(150))

        result = verify_against_transcription(_standard_capo5(confidence=0.9), paths)
        assert result.capo == 5
        assert result.confidence == 0.9
        assert result.source == "audio"
        assert result.verification["checked"] is True
        assert result.verification["veto"] is False
        assert result.verification["subfloor_sustained_count"] == 0

    def test_a_few_subfloor_blips_do_not_veto(self, tmp_path):
        # 3 sustained sub-floor notes + 2 sub-10ms blips: far below every
        # veto threshold — the capo must survive transcription noise.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(100)
        notes += [_note(40, 100.0 + i, dur=0.4) for i in range(3)]
        notes += [_note(38, 110.0 + i, dur=0.02) for i in range(2)]
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.capo == 5
        assert result.verification["veto"] is False
        assert result.verification["subfloor_sustained_count"] == 3

    def test_no_transcription_caps_confidence_and_marks_unverified(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        result = verify_against_transcription(_standard_capo5(confidence=1.0), paths)
        assert result.capo == 5  # behavior kept...
        assert result.confidence <= _UNVERIFIED_CONFIDENCE_CAP  # ...but never 1.0
        assert result.verification["checked"] is False
        assert "unverified" in result.evidence

    def test_voice_program_notes_are_ignored(self, tmp_path):
        # Sub-floor notes tagged as MT3's Singing Voice program (65) are
        # instructor speech, not guitar — they must not trigger the veto.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(100)
        notes += [_note(40, 100.0 + i, dur=0.6, program=65) for i in range(30)]
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.capo == 5
        assert result.verification["veto"] is False
        assert result.verification["subfloor_sustained_count"] == 0

    def test_junk_midi36_cannot_hijack_refit_anchor(self, tmp_path):
        # Regression: 30 genuine sustained E2s plus 3 sustained MT3 bass
        # octave errors at MIDI 36 used to anchor the re-fit at 36 (bare
        # lowest-with-3) and produce "Drop D, half-step down" at confidence
        # 0.0. MIDI 36 is below every candidate's lowest open string, so it
        # must be junk-filtered and the E2s must win: Standard, capo 0.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(100)
        notes += [_note(40, 100.0 + i, dur=0.6) for i in range(30)]  # genuine E2
        notes += [_note(36, 140.0 + i, dur=0.6) for i in range(3)]  # octave junk
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.label == "Standard"
        assert result.capo == 0
        assert result.strings_midi == list(STANDARD_TUNING)
        assert result.confidence == _VETOED_CONFIDENCE_CAP
        assert result.verification["veto"] is True
        assert result.verification["supported_floor_midi"] == 40

    def test_sparse_low_pitch_above_junk_floor_cannot_anchor(self, tmp_path):
        # 3 sustained notes at MIDI 37 (above the junk floor, so not
        # filterable) vs 30 sustained E2s: the 37s carry ~9% of the
        # sub-floor evidence and are far from modal, so the anchor must be
        # the E2s — Standard capo 0, not "Drop D, half-step down".
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(100)
        notes += [_note(40, 100.0 + i, dur=0.6) for i in range(30)]
        notes += [_note(37, 140.0 + i, dur=0.6) for i in range(3)]
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.label == "Standard"
        assert result.capo == 0
        assert result.verification["supported_floor_midi"] == 40

    def test_malformed_mt3_json_falls_back_to_basic_pitch(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.notes_mt3_json.write_text("{not json")
        notes = _body_notes(100) + [_note(40, 100.0 + i, dur=0.6) for i in range(30)]
        paths.notes_json.write_text(json.dumps({"notes": notes}))

        loaded = load_transcribed_notes(paths)
        assert loaded is not None
        assert loaded[1] == "basic_pitch"

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.capo == 0
        assert result.verification["notes_source"] == "basic_pitch"

    def test_non_utf8_mt3_json_falls_back_to_basic_pitch(self, tmp_path):
        # A killed MT3 run can leave truncated non-UTF-8 bytes behind; the
        # UnicodeDecodeError must be swallowed like any other unreadable
        # file, falling back to basic-pitch instead of crashing.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.notes_mt3_json.write_bytes(b"\xff\xfe\x00garbage\x80")
        notes = _body_notes(100) + [_note(40, 100.0 + i, dur=0.6) for i in range(30)]
        paths.notes_json.write_text(json.dumps({"notes": notes}))

        loaded = load_transcribed_notes(paths)
        assert loaded is not None
        assert loaded[1] == "basic_pitch"

        result = verify_against_transcription(_standard_capo5(), paths)
        assert result.capo == 0
        assert result.verification["notes_source"] == "basic_pitch"

    def test_verify_is_idempotent_and_does_not_mutate_input(self, tmp_path):
        # No transcription on disk: the input Tuning must come back
        # untouched, and re-verifying the result must not stack the
        # "capo unverified" suffix or change anything.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        original = _standard_capo5(confidence=1.0)

        first = verify_against_transcription(original, paths)
        assert original.confidence == 1.0
        assert "unverified" not in original.evidence
        assert original.verification is None

        second = verify_against_transcription(first, paths)
        assert second.evidence == first.evidence
        assert second.evidence.count("capo unverified") == 1
        assert second.confidence == first.confidence

    def test_detect_tuning_applies_veto_to_caption_capo(self, tmp_path):
        # End-to-end: captions claim capo 5, transcription is full of E2s —
        # the written tuning.json must carry the vetoed, re-fit result.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.captions_text.write_text("Put your capo on the 5th fret for this one")
        notes = _body_notes(100)
        notes += [_note(40, 100.0 + i, dur=0.6) for i in range(20)]
        notes += [_note(42, 130.0 + i, dur=0.5) for i in range(15)]
        _write_mt3_notes(paths, notes)

        detect_tuning(paths, force=True)
        data = json.loads(paths.tuning_json.read_text())
        assert data["capo"] == 0
        assert data["label"] == "Standard"
        assert data["verification"]["veto"] is True
        assert data["verification"]["subfloor_count"] >= 35


# ---------------------------------------------------------------------------
# Sub-octave-phantom guard
# ---------------------------------------------------------------------------


def _drop_d(label: str = "Drop D") -> Tuning:
    """A dropped-tuning candidate the audio detector might pick from a phantom
    D2 floor (default Drop D; pass 'Double Drop D' for the George case)."""
    return Tuning(
        strings_midi=list(_TUNING_LIBRARY[label]),
        capo=0,
        label=label,
        confidence=0.9,
        source="audio",
        evidence="lowest sustained pitch ≈ MIDI 38.10 (D2)",
    )


def _doubled_floor(pitch: int, partner: int, n: int, start0: float = 50.0) -> list[dict]:
    """n floor notes at ``pitch``, each with a simultaneous note at ``partner``
    (the +12 real note) starting within the co-occurrence window."""
    out: list[dict] = []
    for i in range(n):
        t = start0 + i * 0.7
        out.append(_note(pitch, t, dur=0.5))
        out.append(_note(partner, t + 0.01, dur=0.6))  # +12, ~10ms later
    return out


class TestSubOctavePhantomGuard:
    def test_phantom_d2_floor_rejects_drop_d_to_standard(self, tmp_path):
        # Mirrors S1GpQaD5yT0 (Crystal Ship): every D2(38) floor note is a
        # sub-octave doubling of a simultaneous D3(50). The D2 floor is
        # phantom, so Drop D must flip to Standard (real floor E2=40).
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(120, lo=40, hi=64)  # genuine body incl. E2 floor
        notes += _doubled_floor(38, 50, 30)  # 30 phantom D2s under real D3s
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Standard"
        assert result.capo == 0
        assert result.strings_midi == list(STANDARD_TUNING)
        assert "suboctave-phantom-veto" in result.source
        assert result.confidence <= _VETOED_CONFIDENCE_CAP
        v = result.verification
        assert v["veto"] is True
        assert v["vetoed_label"] == "Drop D"
        assert v["suboctave_floor_count"] == 30
        assert v["suboctave_doubled_fraction"] > 0.8
        assert v["suboctave_phantom_rejected"] is True
        assert v["suboctave_phantom_floor_midi"] == 38
        assert v["refit_floor_midi"] == 40

    def test_real_dropped_low_string_keeps_dropped_tuning(self, tmp_path):
        # Mirrors s_LAzeLbdLs (Song for George): a genuine low D string. Most
        # D2(38) notes stand alone (no +12 partner) — only a minority are
        # doubled — so the doubled fraction stays well under 0.80 and the
        # Double Drop D tuning is KEPT.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(80, lo=40, hi=62)
        # 60 real low-D2s standing alone (no simultaneous D3)...
        notes += [_note(38, 50.0 + i * 0.7, dur=0.5) for i in range(60)]
        # ...plus 20 that happen to coincide with a D3 (~25% doubled).
        notes += _doubled_floor(38, 50, 20, start0=200.0)
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d("Double Drop D"), paths)
        assert result.label == "Double Drop D"
        assert result.strings_midi == _TUNING_LIBRARY["Double Drop D"]
        assert result.source == "audio"
        v = result.verification
        assert v["veto"] is False
        assert v["suboctave_floor_count"] == 80
        assert v["suboctave_doubled_fraction"] < 0.5
        assert v["suboctave_phantom_rejected"] is False

    def test_standard_floor_at_or_above_midi40_is_never_inspected(self, tmp_path):
        # A Standard (floor E2=40) result whose E2 notes are heavily doubled by
        # E3(52). The guard is INERT at/above MIDI 40 (protects LBTD/Angie):
        # no suboctave_* fields, tuning untouched.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(80, lo=45, hi=64)
        notes += _doubled_floor(40, 52, 40)  # heavily-doubled E2 floor
        _write_mt3_notes(paths, notes)

        standard = Tuning(
            strings_midi=list(STANDARD_TUNING),
            capo=0,
            label="Standard",
            confidence=0.9,
            source="audio",
            evidence="lowest sustained pitch ≈ MIDI 40.0 (E2)",
        )
        result = verify_against_transcription(standard, paths)
        assert result.label == "Standard"
        assert result.capo == 0
        assert result.source == "audio"
        assert result.verification["veto"] is False
        assert "suboctave_floor_count" not in result.verification
        assert "suboctave_phantom_rejected" not in result.verification

    def test_too_few_floor_notes_does_not_reject(self, tmp_path):
        # Below _MIN_FLOOR_NOTES phantom-doubled D2s: even at 100% doubled the
        # sample is too small to conclude anything, so Drop D is KEPT.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        few = _MIN_FLOOR_NOTES - 1
        notes = _body_notes(80, lo=40, hi=64)
        notes += _doubled_floor(38, 50, few)
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Drop D"
        assert result.strings_midi == _TUNING_LIBRARY["Drop D"]
        v = result.verification
        assert v["veto"] is False
        assert v["suboctave_floor_count"] == few
        assert v["suboctave_doubled_fraction"] > 0.8  # high fraction, but too few notes
        assert v["suboctave_phantom_rejected"] is False

    def test_heavily_doubled_real_floor_is_not_skipped_in_refit(self, tmp_path):
        # Major-finding guard: a phantom D2(38) floor sits beneath a REAL but
        # heavily octave-doubled E2(40) low string (E2 struck inside E2+E3 power
        # chords reads >80% doubled too). The bounded phantom-skip in the re-fit
        # must NOT skip E2 — it is at phantom_floor + _REFIT_MAX_SKIP_SEMITONES,
        # exactly the boundary — so the re-fit lands on Standard (floor 40),
        # NOT an absurd 'Standard, capo N'.
        assert 38 + _REFIT_MAX_SKIP_SEMITONES == 40  # E2 sits at the skip boundary
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        # Body at/above A2 so the ONLY low strings are the phantom D2 and the
        # heavily-doubled real E2.
        notes = _body_notes(120, lo=45, hi=64)
        notes += _doubled_floor(38, 50, 30)  # 30 phantom D2s under real D3s
        notes += _doubled_floor(40, 52, 30, start0=300.0)  # 30 doubled real E2s
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Standard"
        assert result.capo == 0
        assert result.strings_midi == list(STANDARD_TUNING)
        assert "suboctave-phantom-veto" in result.source
        v = result.verification
        assert v["veto"] is True
        assert v["suboctave_phantom_floor_midi"] == 38
        assert v["refit_floor_midi"] == 40  # the real E2, not skipped past

    def test_phantom_floor_refits_to_capoed_standard(self, tmp_path):
        # A phantom D2(38) floor whose real low string is A2(45): nothing
        # supported sits between, so the re-fit lands on 'Standard, capo 5'
        # (effective floor 45). Exercises the capo re-introduction path.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(120, lo=45, hi=64)  # real floor A2=45, nothing below
        notes += _doubled_floor(38, 50, 30)  # phantom D2s under real D3s
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Standard, capo 5"
        assert result.capo == 5
        assert result.strings_midi == list(STANDARD_TUNING)
        assert "suboctave-phantom-veto" in result.source
        v = result.verification
        assert v["veto"] is True
        assert v["suboctave_phantom_floor_midi"] == 38
        assert v["refit_floor_midi"] == 45

    def test_phantom_floor_with_no_low_support_falls_back_to_standard_e(self, tmp_path):
        # Degenerate: ONLY phantom D2(38)s under real D3(50)s, and NOTHING else
        # at all above the phantom floor (not even the D3 partner sustains long
        # enough to count). With no supported pitch above the phantom floor the
        # re-fit falls back to STANDARD low E (40) — the documented fallback.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        # Phantom D2s whose D3 partners are SHORTER than _SUBFLOOR_MIN_NOTE_S so
        # they do not register as sustained support above the phantom floor.
        notes: list[dict] = []
        for i in range(30):
            t = 50.0 + i * 0.7
            notes.append(_note(38, t, dur=0.5))  # phantom D2 (sustained)
            notes.append(_note(50, t + 0.01, dur=0.05))  # D3 partner, too short
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Standard"
        assert result.capo == 0
        assert result.strings_midi == list(STANDARD_TUNING)
        assert "suboctave-phantom-veto" in result.source
        v = result.verification
        assert v["veto"] is True
        assert v["suboctave_phantom_floor_midi"] == 38
        assert v["refit_floor_midi"] == STANDARD_TUNING[0]  # 40

    def test_phantom_floor_only_partner_above_climbs_to_capoed_standard(self, tmp_path):
        # KNOWN, NON-BLOCKING limitation (review major finding): when the ONLY
        # sustained support above a phantom D2(38) floor is the phantom's own
        # +12 partner D3(50) — i.e. there is no genuine low-register string at
        # all — the re-fit anchors on D3=50 and climbs to 'Standard, capo 7'
        # rather than the standard-E fallback. This does not occur in the four
        # target songs (Crystal lands on E2=40). Locked in here so any future
        # change to the fallback semantics is a deliberate, visible decision.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _doubled_floor(38, 50, 30)  # phantom D2s + SUSTAINED D3 partners
        _write_mt3_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Standard, capo 7"
        assert result.capo == 7
        v = result.verification
        assert v["veto"] is True
        assert v["suboctave_phantom_floor_midi"] == 38
        # Anchored on the D3(50) partner; capo maxes at 7 so the effective floor
        # lands at 40+7=47 (closest reachable to 50).
        assert v["refit_floor_midi"] == 47

    def test_phantom_guard_uses_basic_pitch_source(self, tmp_path):
        # The phantom path must work when the only transcription is basic-pitch
        # (notes.json), not mt3. The evidence string and verification must cite
        # 'basic_pitch' as the notes_source.
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        notes = _body_notes(120, lo=40, hi=64)  # genuine body incl. E2 floor
        notes += _doubled_floor(38, 50, 30)  # 30 phantom D2s under real D3s
        _write_basic_pitch_notes(paths, notes)

        result = verify_against_transcription(_drop_d(), paths)
        assert result.label == "Standard"
        assert result.strings_midi == list(STANDARD_TUNING)
        assert "suboctave-phantom-veto" in result.source
        assert "basic_pitch" in result.evidence
        v = result.verification
        assert v["veto"] is True
        assert v["notes_source"] == "basic_pitch"
        assert v["refit_floor_midi"] == 40
