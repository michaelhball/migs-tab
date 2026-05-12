"""Tests for tuning.py — caption pattern matching, library defaults, audio
candidate generation."""

from __future__ import annotations

import json

from migs_tab.paths import VideoPaths
from migs_tab.tuning import (
    _AUDIO_CANDIDATES,
    _TUNING_LIBRARY,
    STANDARD_TUNING,
    Tuning,
    _detect_from_captions,
    default_tuning,
    detect_tuning,
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
