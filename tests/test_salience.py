"""Tests for salience.py — synthetic-audio salience, octave flags, velocity.

All synthetic tests are seeded and fast. Tests that need the local cache
(real Angie stem) are skipped when cache/ is absent (CI has no cache).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from migs_tab.salience import (
    DEFAULT_SR,
    OCTAVE_ARTIFACT_RATIO,
    PITCH_MAX_MIDI,
    PITCH_MIN_MIDI,
    PSEUDO_VELOCITY_UNSCORED,
    compare_pitches_at_onset,
    compute_cqt_context,
    karplus_strong_render,
    load_stem_window,
    note_salience,
    octave_artifact_flags,
    pseudo_velocity,
    section_score,
)

_CACHE_STEM = Path(__file__).resolve().parents[1] / "cache" / "wS_i91qxQYM" / "stems" / "other.wav"


def _sine(midi_pitch: int, dur: float, sr: int = DEFAULT_SR, amp: float = 0.5) -> np.ndarray:
    f0 = librosa.midi_to_hz(midi_pitch)
    t = np.arange(int(dur * sr)) / sr
    y = amp * np.sin(2 * np.pi * f0 * t)
    # fade edges to avoid spectral splatter from clicks
    n = min(128, len(y))
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def _place(mix: np.ndarray, y: np.ndarray, onset: float, sr: int = DEFAULT_SR) -> None:
    i0 = int(onset * sr)
    mix[i0 : i0 + len(y)] += y


class TestNoteSalience:
    def test_true_pitch_outranks_neighbors(self):
        # Three KS-rendered notes at known pitches/onsets.
        events = [(0.2, 0.5, 52), (1.0, 0.5, 57), (1.8, 0.5, 64)]
        y = karplus_strong_render(events, sr=DEFAULT_SR, seed=11)
        queries = [(t, p) for t, _, p in events]
        true_scores = note_salience(queries, y, DEFAULT_SR, window_start=0.0)
        for delta in (-2, -1, 1, 2):
            shifted = [(t, p + delta) for t, p in queries]
            shifted_scores = note_salience(shifted, y, DEFAULT_SR, window_start=0.0)
            for s_true, s_shift in zip(true_scores, shifted_scores, strict=True):
                assert s_true is not None and s_shift is not None
                assert s_true > s_shift

    def test_absolute_timestamps_respect_window_start(self):
        # Same audio, declared to start at t=100s; absolute onsets must work.
        y = karplus_strong_render([(0.3, 0.5, 60)], sr=DEFAULT_SR, seed=3)
        [score] = note_salience([(100.3, 60)], y, DEFAULT_SR, window_start=100.0)
        assert score is not None and score > 0.95

    def test_event_outside_window_returns_none(self):
        y = _sine(60, 1.0)
        before, after = note_salience([(-5.0, 60), (50.0, 60)], y, DEFAULT_SR, window_start=0.0)
        assert before is None
        assert after is None

    def test_pitch_outside_cqt_range_returns_none(self):
        y = _sine(60, 1.0)
        low, high = note_salience(
            [(0.1, PITCH_MIN_MIDI - 1), (0.1, PITCH_MAX_MIDI + 1)],
            y,
            DEFAULT_SR,
            window_start=0.0,
        )
        assert low is None
        assert high is None

    def test_onset_at_window_edge_clips_gracefully(self):
        # Onset so close to the end that only part of the 30-200ms window
        # has frames — must score from what exists, not raise.
        y = _sine(60, 1.0)
        [score] = note_salience([(0.95, 60)], y, DEFAULT_SR, window_start=0.0)
        assert score is None or 0.0 < score <= 1.0

    def test_silence_returns_none_not_perfect_score(self):
        # All-zero audio (zero-padded tails, loads past EOF, silent stem
        # gaps): an all-zero post-onset column would rank EVERY pitch a
        # perfect 1.0 via (col <= 0).mean() — must be None, not "verified".
        y = np.zeros(int(1.0 * DEFAULT_SR))
        scores = note_salience([(0.1, 57), (0.5, 64)], y, DEFAULT_SR, window_start=0.0)
        assert scores == [None, None]


class TestComparePitchesAtOnset:
    def test_true_pitch_ranked_first(self):
        y = karplus_strong_render([(0.2, 0.6, 57)], sr=DEFAULT_SR, seed=5)
        ctx = compute_cqt_context(y, DEFAULT_SR, window_start=0.0)
        ranked = compare_pitches_at_onset(0.2, [55, 56, 57, 58, 59], ctx)
        assert ranked[0][0] == 57
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_out_of_range_candidate_ranks_last_with_zero(self):
        y = _sine(60, 1.0)
        ctx = compute_cqt_context(y, DEFAULT_SR, window_start=0.0)
        ranked = compare_pitches_at_onset(0.1, [60, PITCH_MAX_MIDI + 5], ctx)
        assert ranked[0][0] == 60
        assert ranked[-1] == (PITCH_MAX_MIDI + 5, 0.0)

    def test_all_silent_candidates_score_zero(self):
        # Over digital silence no candidate may win with a fake 1.0; all
        # are unscorable (None from salience_at) and collapse to 0.0.
        ctx = compute_cqt_context(np.zeros(int(1.0 * DEFAULT_SR)), DEFAULT_SR, window_start=0.0)
        ranked = compare_pitches_at_onset(0.1, [55, 57, 59], ctx)
        assert [s for _, s in ranked] == [0.0, 0.0, 0.0]


class TestSharedCQTContext:
    def test_prebuilt_ctx_matches_internal_build(self):
        # fret.py's pre-Viterbi integration shares one CQT across scoring
        # functions; passing it must be a pure optimization, never a
        # behavior change.
        events = [(0.2, 0.5, 52), (1.0, 0.5, 57), (1.8, 0.5, 64)]
        y = karplus_strong_render(events, sr=DEFAULT_SR, seed=11)
        queries = [(t, p) for t, _, p in events]
        ctx = compute_cqt_context(y, DEFAULT_SR, window_start=0.0)
        assert note_salience(queries, y, DEFAULT_SR, 0.0, ctx=ctx) == note_salience(
            queries, y, DEFAULT_SR, 0.0
        )
        assert pseudo_velocity(queries, y, DEFAULT_SR, 0.0, ctx=ctx) == pseudo_velocity(
            queries, y, DEFAULT_SR, 0.0
        )


class TestOctaveArtifactFlags:
    def test_quiet_overtone_flagged_real_octave_not(self):
        sr = DEFAULT_SR
        mix = np.zeros(int(1.5 * sr))
        # Cluster A at t=0.1: loud A3 + its 2nd harmonic at 5% amplitude —
        # a phantom A4 "note" claimed on top of pure overtone energy.
        _place(mix, _sine(57, 1.0, sr, amp=0.8), 0.1, sr)
        _place(mix, _sine(69, 1.0, sr, amp=0.04), 0.1, sr)
        ctx = compute_cqt_context(mix, sr, window_start=0.0)
        flags = octave_artifact_flags([(0.1, 57), (0.1, 69)], ctx)
        assert flags == [False, True]

        # Cluster B: genuinely-played octave pair at comparable energy.
        mix2 = np.zeros(int(1.5 * sr))
        _place(mix2, _sine(57, 1.0, sr, amp=0.7), 0.1, sr)
        _place(mix2, _sine(69, 1.0, sr, amp=0.6), 0.1, sr)
        ctx2 = compute_cqt_context(mix2, sr, window_start=0.0)
        flags2 = octave_artifact_flags([(0.1, 57), (0.1, 69)], ctx2)
        assert flags2 == [False, False]

    def test_no_partner_never_flagged(self):
        y = _sine(69, 1.0, amp=0.01)  # very quiet, but no 57 in the cluster
        ctx = compute_cqt_context(y, DEFAULT_SR, window_start=0.0)
        assert octave_artifact_flags([(0.1, 69), (0.1, 64)], ctx) == [False, False]

    def test_threshold_is_calibrated_not_guessed(self):
        # Guard the calibrated constant against drive-by edits: it must sit
        # below the measured genuine-pair p5 (0.242) and above the proven
        # B4-phantom maximum (0.130). See the constant's comment in salience.py.
        assert 0.130 < OCTAVE_ARTIFACT_RATIO <= 0.242

    def test_out_of_window_event_not_flagged(self):
        y = _sine(57, 1.0)
        ctx = compute_cqt_context(y, DEFAULT_SR, window_start=0.0)
        assert octave_artifact_flags([(99.0, 57), (99.0, 69)], ctx) == [False, False]


class TestPseudoVelocity:
    def test_monotone_in_amplitude(self):
        sr = DEFAULT_SR
        mix = np.zeros(int(3.5 * sr))
        amps = [0.05, 0.2, 0.8]
        onsets = [0.2, 1.3, 2.4]
        for onset, amp in zip(onsets, amps, strict=True):
            _place(mix, _sine(57, 0.8, sr, amp=amp), onset, sr)
        vels = pseudo_velocity([(t, 57) for t in onsets], mix, sr, window_start=0.0)
        assert all(0 <= v <= 127 for v in vels)
        assert vels[0] < vels[1] < vels[2]

    def test_unscorable_event_gets_neutral_velocity(self):
        y = _sine(60, 1.0)
        vels = pseudo_velocity(
            [(50.0, 60), (0.1, PITCH_MIN_MIDI - 4)], y, DEFAULT_SR, window_start=0.0
        )
        assert vels == [PSEUDO_VELOCITY_UNSCORED, PSEUDO_VELOCITY_UNSCORED]

    def test_silence_scores_zero(self):
        sr = DEFAULT_SR
        mix = np.zeros(int(2.0 * sr))
        _place(mix, _sine(57, 0.5, sr, amp=0.8), 0.1, sr)  # something audible
        # ...but query a silent stretch long after the note has ended.
        [vel] = pseudo_velocity([(1.5, 80)], mix, sr, window_start=0.0)
        assert vel < 30


class TestKarplusStrongRender:
    def test_deterministic_for_seed(self):
        events = [(0.1, 0.5, 52), (0.7, 0.5, 57)]
        a = karplus_strong_render(events, seed=7)
        b = karplus_strong_render(events, seed=7)
        assert np.array_equal(a, b)

    def test_empty_events(self):
        assert len(karplus_strong_render([])) == 0

    def test_peak_normalized(self):
        y = karplus_strong_render([(0.0, 0.5, 57)], seed=1)
        assert np.max(np.abs(y)) == pytest.approx(0.9, abs=1e-6)

    def test_short_final_note_rings_full_min_duration(self):
        # A 0.05s final note's pluck is clamped up to _KS_MIN_DUR (0.4s);
        # the render length must honor the same clamp, not truncate the
        # still-ringing tail at d + _KS_RING_EXTRA (0.3s).
        y = karplus_strong_render([(0.0, 0.05, 57)], sr=DEFAULT_SR, seed=1)
        assert len(y) == int(0.4 * DEFAULT_SR)


class TestSectionScore:
    def test_true_events_beat_shifted_events(self, tmp_path):
        sr = DEFAULT_SR
        events = [
            (0.2, 0.4, 45),
            (0.7, 0.4, 52),
            (1.2, 0.4, 57),
            (1.7, 0.4, 60),
            (2.2, 0.4, 64),
        ]
        # "Real" stem = KS render with a different seed than section_score
        # uses internally, so we're not comparing identical noise.
        stem = karplus_strong_render(events, sr=sr, seed=99)
        stem_path = tmp_path / "stem.wav"
        sf.write(stem_path, stem.astype(np.float32), sr)

        good = section_score(events, stem_path, 0.0, 3.0, sr=sr)
        shifted = section_score([(t, d, p + 5) for t, d, p in events], stem_path, 0.0, 3.0, sr=sr)
        assert good > shifted
        assert good > 0.8

    def test_no_events_in_window_scores_zero(self, tmp_path):
        sr = DEFAULT_SR
        stem_path = tmp_path / "stem.wav"
        sf.write(stem_path, _sine(57, 2.0).astype(np.float32), sr)
        assert section_score([(50.0, 0.5, 57)], stem_path, 0.0, 2.0, sr=sr) == 0.0


class TestLoadStemWindow:
    def test_loads_requested_slice_only(self, tmp_path):
        sr = DEFAULT_SR
        stem_path = tmp_path / "stem.wav"
        sf.write(stem_path, _sine(57, 5.0).astype(np.float32), sr)
        y, sr_out = load_stem_window(stem_path, 1.0, 3.0, sr=sr)
        assert sr_out == sr
        assert len(y) == pytest.approx(2.0 * sr, rel=0.01)

    def test_negative_start_raises(self, tmp_path):
        sr = DEFAULT_SR
        stem_path = tmp_path / "stem.wav"
        sf.write(stem_path, _sine(57, 2.0).astype(np.float32), sr)
        with pytest.raises(ValueError, match="start"):
            load_stem_window(stem_path, -1.0, 1.0, sr=sr)


@pytest.mark.skipif(not _CACHE_STEM.exists(), reason="local cache not available")
class TestAgainstCachedAngie:
    """Spot-checks against the proven phantom overtones in the Angie cache."""

    def test_e7_cluster_phantom_b4_flagged(self):
        # Open-E7 strums at 903.2-904.8s: B4=71 is a proven phantom overtone
        # of the (loud, real) B3=59; measured E(71)/E(59) = 0.016-0.130.
        y, sr = load_stem_window(_CACHE_STEM, 901.0, 907.0)
        ctx = compute_cqt_context(y, sr, window_start=901.0)
        cluster = [(903.54, 40), (903.54, 59), (903.54, 62), (903.54, 64), (903.54, 71)]
        flags = octave_artifact_flags(cluster, ctx)
        assert flags == [False, False, False, False, True]

    def test_verified_window_salience_is_high(self):
        # Chroma-verified-good window: tab notes should mostly be the
        # strongest thing ringing (measured mean 0.960 over 1107-1125s).
        import json

        frets = json.loads((_CACHE_STEM.parents[1] / "frets.json").read_text())
        tuning = frets["tuning"]["low_to_high_midi"]
        events = [
            (n["start"], tuning[n["string"]] + n["fret"])
            for n in frets["notes"]
            if 1107 <= n["start"] < 1113
        ]
        y, sr = load_stem_window(_CACHE_STEM, 1106.0, 1114.0)
        scores = [s for s in note_salience(events, y, sr, window_start=1106.0) if s is not None]
        assert len(scores) >= 10
        assert float(np.mean(scores)) > 0.85
