"""Tests for the basic-pitch → ornament-hint annotation pipeline."""

from __future__ import annotations

from migs_tab.annotations import (
    _format_times,
    compute_section_hints,
    midi_to_name,
)


def _n(start: float, pitch: int, end: float | None = None) -> dict:
    return {"start": start, "pitch": pitch, "end": end if end is not None else start + 0.1}


class TestMidiToName:
    def test_middle_c(self):
        assert midi_to_name(60) == "C4"

    def test_high_e(self):
        # 12th-fret natural harmonic on E string = E5 = pitch 76 (one octave above E4=64).
        assert midi_to_name(76) == "E5"

    def test_low_e(self):
        assert midi_to_name(40) == "E2"


class TestFormatTimes:
    def test_single(self):
        assert _format_times([1.2]) == "+1.2s"

    def test_three(self):
        assert _format_times([1.0, 2.5, 3.0]) == "+1.0s, +2.5s, +3.0s"

    def test_truncated(self):
        s = _format_times([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert s.startswith("+1.0s, +2.0s, …")
        assert "(+4 more)" in s


class TestComputeSectionHints:
    def test_no_secondary_notes(self):
        assert compute_section_hints(0.0, 10.0, [_n(1.0, 60)], []) == []

    def test_exact_pitch_match_within_tol(self):
        """basic-pitch note matching MT3 note → no hint."""
        primary = [_n(1.0, 60)]
        secondary = [_n(1.05, 60)]  # 50ms off, within 100ms tol
        assert compute_section_hints(0.0, 10.0, primary, secondary) == []

    def test_harmonic_candidate_with_fundamental(self):
        """High pitch in secondary + fundamental in primary → harmonic hint."""
        primary = [_n(1.0, 64)]  # E4 — the fundamental
        secondary = [_n(1.0, 76)]  # E5 — one octave up
        hints = compute_section_hints(0.0, 10.0, primary, secondary)
        assert len(hints) == 1
        assert "natural harmonic candidate" in hints[0]
        assert "E5" in hints[0]
        assert "E4" in hints[0]

    def test_high_pitch_without_fundamental_needs_repetition(self):
        """High pitch in secondary with no fundamental nearby → only hint if ≥2 hits."""
        primary = [_n(1.0, 50)]  # different pitch, no octave relation
        secondary_single = [_n(1.0, 80)]
        secondary_multi = [_n(1.0, 80), _n(2.0, 80)]

        assert compute_section_hints(0.0, 10.0, primary, secondary_single) == []

        hints = compute_section_hints(0.0, 10.0, primary, secondary_multi)
        assert len(hints) == 1
        assert "high lead" in hints[0]

    def test_sustained_low_pitch_needs_repetition_and_duration(self):
        primary: list[dict] = []
        # Two sustained instances → hint fires.
        secondary = [_n(1.0, 60, end=1.3), _n(3.0, 60, end=3.5)]
        hints = compute_section_hints(0.0, 10.0, primary, secondary)
        assert len(hints) == 1
        assert "sustained" in hints[0]

    def test_short_notes_below_threshold_skipped(self):
        primary: list[dict] = []
        # Same pitch, but both notes are <200ms → no hint.
        secondary = [_n(1.0, 60, end=1.05), _n(3.0, 60, end=3.05)]
        assert compute_section_hints(0.0, 10.0, primary, secondary) == []

    def test_windowing(self):
        """Notes outside [start, end) are ignored."""
        primary: list[dict] = []
        # Both notes are in the window, sustained → would normally hint.
        secondary = [_n(1.0, 60, end=1.3), _n(2.0, 60, end=2.3)]
        # Window 5-10 excludes both → no hints.
        assert compute_section_hints(5.0, 10.0, primary, secondary) == []

    def test_harmonic_dedupes_by_pitch(self):
        """Multiple harmonic candidates at the same pitch collapse into one hint."""
        primary = [_n(1.0, 64), _n(2.0, 64), _n(3.0, 64)]
        secondary = [_n(1.0, 76), _n(2.0, 76), _n(3.0, 76)]
        hints = compute_section_hints(0.0, 10.0, primary, secondary)
        # One hint, listing all three times.
        assert len(hints) == 1
        assert "+1.0s" in hints[0] and "+2.0s" in hints[0] and "+3.0s" in hints[0]

    def test_sustained_skipped_when_pitch_in_primary_elsewhere(self):
        """If MT3 has the same pitch anywhere in the section, the sustained
        secondary version is a chord tone, not a missed ornament."""
        # MT3 has pitch 60 at t=5s — outside the ±100ms onset-match window
        # of the secondary notes, so the secondary notes ARE "novel" by the
        # narrow definition. But the pitch IS in the primary, so we skip.
        primary = [_n(5.0, 60)]
        secondary = [_n(1.0, 60, end=1.3), _n(3.0, 60, end=3.3)]
        assert compute_section_hints(0.0, 10.0, primary, secondary) == []

    def test_sustained_still_fires_when_pitch_absent_from_primary(self):
        """Sustained hint should still fire when MT3 truly has no note at this pitch."""
        primary = [_n(1.0, 50)]  # different pitch
        secondary = [_n(1.0, 60, end=1.3), _n(3.0, 60, end=3.3)]
        hints = compute_section_hints(0.0, 10.0, primary, secondary)
        assert len(hints) == 1
        assert "sustained" in hints[0]

    def test_harmonic_priority_over_sustained(self):
        """Harmonic candidates rank before sustained hints when capped."""
        primary = [_n(1.0, 64)]
        secondary = [
            _n(1.0, 76),  # harmonic candidate (E5 over E4)
            _n(2.0, 50, end=2.4),  # sustained
            _n(3.0, 50, end=3.4),
        ]
        hints = compute_section_hints(0.0, 10.0, primary, secondary)
        assert "harmonic" in hints[0]
