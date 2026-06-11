"""Tests for verify.py — per-note verdicts, cross-model matching, capo
report passthrough, section banding, and the verification.json schema.

Synthetic fixtures only (tmp_path cache + short synthetic stem WAV); tests
that need the real local cache are skipped when cache/ is absent (CI).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from migs_tab import verify as verify_mod
from migs_tab.paths import VideoPaths
from migs_tab.salience import (
    DEFAULT_SR,
    PITCH_MAX_MIDI,
    PITCH_MIN_MIDI,
    CQTContext,
    karplus_strong_render,
    load_stem_window,
)
from migs_tab.verify import (
    AGREEMENT_BOTH,
    AGREEMENT_MT3_ONLY,
    BAND_BAD,
    BAND_NO_NOTES,
    BAND_SOLID,
    BAND_SUSPECT,
    SECTION_BAND_BAD_MAX,
    SECTION_BAND_SOLID_MIN,
    VERDICT_OCTAVE,
    VERDICT_PHANTOM,
    VERDICT_SUPPORTED,
    VERDICT_UNSCORED,
    VERDICT_WEAK,
    VERDICTS,
    VerifyError,
    _chunk_indices,
    _load_notes_list,
    _note_pitch,
    assess_note,
    band_for_score,
    capo_check,
    match_events,
    verify,
)

_CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"
_ANGIE_STEM = _CACHE_ROOT / "wS_i91qxQYM" / "stems" / "other.wav"
_LBTD_TUNING = _CACHE_ROOT / "9jswOBilMvA" / "tuning.json"

_N_BINS = PITCH_MAX_MIDI - PITCH_MIN_MIDI + 1


# ---------------------------------------------------------------------------
# Handcrafted-CQT unit tests for the verdict tiers
# ---------------------------------------------------------------------------


def _ctx_from_column(col: np.ndarray) -> CQTContext:
    """A CQTContext whose every frame holds ``col`` — onset 0.0 is scorable."""
    mags = np.tile(np.asarray(col, dtype=float)[:, None], (1, 5))
    frame_times = np.linspace(0.05, 0.15, 5)  # inside the 30-200 ms window
    return CQTContext(magnitudes=mags, frame_times=frame_times, window_start=0.0)


def _bin(pitch: int) -> int:
    return pitch - PITCH_MIN_MIDI


class TestAssessNote:
    def test_supported_when_claimed_bin_is_loudest(self):
        col = np.full(_N_BINS, 0.001)
        col[_bin(60)] = 1.0
        s, verdict = assess_note(_ctx_from_column(col), 0.0, 60)
        assert s == 1.0
        assert verdict == VERDICT_SUPPORTED

    def test_weak_tier_at_intermediate_rank(self):
        # Ascending values by bin: claimed bin ranks 42/58 = 0.724 → weak.
        # Its +12 partner ranks higher but at energy ratio 42/54 = 0.78 the
        # strict octave rule must not fire.
        col = np.arange(1.0, _N_BINS + 1)
        pitch = PITCH_MIN_MIDI + 41
        s, verdict = assess_note(_ctx_from_column(col), 0.0, pitch)
        assert s == pytest.approx(42 / 58)
        assert verdict == VERDICT_WEAK

    def test_phantom_suspect_at_low_rank(self):
        col = np.arange(1.0, _N_BINS + 1)
        pitch = PITCH_MIN_MIDI + 10  # rank 11/58 = 0.19; p-12 out of range
        s, verdict = assess_note(_ctx_from_column(col), 0.0, pitch)
        assert s == pytest.approx(11 / 58)
        assert verdict == VERDICT_PHANTOM

    def test_octave_suspect_strict_rule(self):
        # E(p)/E(p-12) = 0.1 < 0.20 with the partner out-ranking p.
        col = np.full(_N_BINS, 0.001)
        col[_bin(72)] = 0.1
        col[_bin(60)] = 1.0
        s, verdict = assess_note(_ctx_from_column(col), 0.0, 72)
        assert verdict == VERDICT_OCTAVE
        assert s is not None

    def test_octave_suspect_loose_rule_overrides_supported(self):
        # Ratio 0.5 escapes the strict rule, but the p-12 partner is
        # top-ranked and the claimed note trails by > the loose margin —
        # the A5=81-style phantom from the Angie calibration.
        col = np.full(_N_BINS, 0.001)
        col[_bin(72)] = 0.5
        col[_bin(60)] = 1.0
        for b in (_bin(40), _bin(43), _bin(46), _bin(49)):  # away from 72±12
            col[b] = 0.7
        s, verdict = assess_note(_ctx_from_column(col), 0.0, 72)
        assert s >= verify_mod.SALIENCE_SUPPORTED_MIN  # tier alone says supported
        assert verdict == VERDICT_OCTAVE

    def test_unscored_on_silent_bin(self):
        col = np.full(_N_BINS, 0.5)
        col[_bin(60)] = 0.0
        s, verdict = assess_note(_ctx_from_column(col), 0.0, 60)
        assert s is None
        assert verdict == VERDICT_UNSCORED

    def test_unscored_outside_window_and_range(self):
        ctx = _ctx_from_column(np.full(_N_BINS, 0.5))
        assert assess_note(ctx, 50.0, 60) == (None, VERDICT_UNSCORED)
        assert assess_note(ctx, 0.0, PITCH_MAX_MIDI + 1) == (None, VERDICT_UNSCORED)


class TestBands:
    def test_band_boundaries(self):
        assert band_for_score(SECTION_BAND_SOLID_MIN) == BAND_SOLID
        assert band_for_score(SECTION_BAND_SOLID_MIN - 0.001) == BAND_SUSPECT
        assert band_for_score(SECTION_BAND_BAD_MAX) == BAND_SUSPECT
        assert band_for_score(SECTION_BAND_BAD_MAX - 0.001) == BAND_BAD


class TestMatchEvents:
    def test_within_tolerance_matches(self):
        flags, unmatched = match_events([(1.0, 60)], [(1.08, 60)])
        assert flags == [True]
        assert unmatched == 0

    def test_outside_tolerance_or_wrong_pitch_does_not(self):
        flags, unmatched = match_events([(1.0, 60), (2.0, 60)], [(1.11, 60), (2.0, 61)])
        assert flags == [False, False]
        assert unmatched == 2

    def test_each_secondary_consumed_once(self):
        flags, unmatched = match_events([(1.0, 60), (1.05, 60)], [(1.02, 60)])
        assert sum(flags) == 1
        assert unmatched == 0

    def test_greedy_prefers_nearest_onset(self):
        flags, unmatched = match_events([(1.0, 60)], [(0.95, 60), (1.01, 60)])
        assert flags == [True]
        assert unmatched == 1

    def test_exact_tolerance_boundary_is_symmetric(self):
        # Candidates at EXACTLY tol must match in both directions (tol=0.5 is
        # exactly representable, unlike the default 0.100 boundary).
        fwd, _ = match_events([(1.0, 60)], [(1.5, 60)], tol=0.5)
        bwd, _ = match_events([(1.5, 60)], [(1.0, 60)], tol=0.5)
        assert fwd == [True]
        assert bwd == [True]


class TestChunking:
    def test_splits_on_silence_gap_and_span(self):
        onsets = [0.0, 5.0, 9.0, 100.0, 105.0, 170.0]
        chunks = _chunk_indices(onsets)
        # silence gap 9→100 splits; span 100→170 exceeds 60 s and splits again.
        assert chunks == [[0, 1, 2], [3, 4], [5]]

    def test_indices_align_with_unsorted_input(self):
        chunks = _chunk_indices([8.0, 0.0])
        assert chunks == [[1, 0]]


class TestNotePitch:
    def test_pitch_field_wins(self):
        assert _note_pitch({"pitch": 57, "string": 0, "fret": 0}, [40, 45, 50, 55, 59, 64]) == 57

    def test_derived_from_tuning_string_fret(self):
        assert _note_pitch({"string": 3, "fret": 2}, [40, 45, 50, 55, 59, 64]) == 57

    def test_underivable_returns_none(self):
        assert _note_pitch({"string": 3, "fret": 2}, None) is None
        assert _note_pitch({"fret": 2}, [40, 45, 50, 55, 59, 64]) is None

    def test_out_of_range_string_returns_none(self):
        # A negative index must not wrap (Python indexing) into a wrong pitch.
        tuning = [40, 45, 50, 55, 59, 64]
        assert _note_pitch({"string": -1, "fret": 2}, tuning) is None
        assert _note_pitch({"string": 6, "fret": 2}, tuning) is None


class TestLoadNotesList:
    def test_missing_or_unusable_returns_none(self, tmp_path):
        assert _load_notes_list(tmp_path / "absent.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert _load_notes_list(bad) is None

    def test_present_but_empty_returns_empty_list(self, tmp_path):
        # Distinguished from missing so verify can report the right reason.
        p = tmp_path / "notes.json"
        p.write_text(json.dumps({"notes": []}))
        assert _load_notes_list(p) == []


# ---------------------------------------------------------------------------
# Integration: synthetic cache fixture end-to-end
# ---------------------------------------------------------------------------

_VID = "testverify0"

# Three clean Karplus-Strong notes; the stem in [0, 5) is exactly what
# section_score will synthesize for them (same renderer, same default seed),
# so the "clean" section must self-match into the solid band.
_CLEAN_EVENTS = [(0.5, 1.0, 48), (2.0, 1.0, 60), (3.5, 1.0, 64)]
_SINE_PITCH = 57  # A3 sounds 6.0-8.5 s
_STEM_DUR_S = 9.0


def _sine(midi_pitch: int, dur: float, sr: int = DEFAULT_SR, amp: float = 0.5) -> np.ndarray:
    f0 = 440.0 * 2 ** ((midi_pitch - 69) / 12)
    t = np.arange(int(dur * sr)) / sr
    y = amp * np.sin(2 * np.pi * f0 * t)
    n = min(128, len(y))
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


def _fixture_paths(tmp_path: Path) -> VideoPaths:
    paths = VideoPaths(_VID, cache_dir=tmp_path)
    paths.stems_dir.mkdir(parents=True, exist_ok=True)

    mix = np.zeros(int(_STEM_DUR_S * DEFAULT_SR))
    ks = karplus_strong_render(_CLEAN_EVENTS, sr=DEFAULT_SR)
    mix[: len(ks)] += ks
    sine = _sine(_SINE_PITCH, 2.5)
    mix[int(6.0 * DEFAULT_SR) : int(6.0 * DEFAULT_SR) + len(sine)] += sine
    sf.write(paths.guitar_stem, mix, DEFAULT_SR)

    notes = [
        # 0-2: the clean KS notes → supported.
        *(
            {"note_index": i, "start": t, "end": t + d, "pitch": p, "cluster_id": i}
            for i, (t, d, p) in enumerate(_CLEAN_EVENTS)
        ),
        # 3: the real sine → supported.
        {"note_index": 3, "start": 6.0, "end": 7.0, "pitch": _SINE_PITCH, "cluster_id": 3},
        # 4: +12 octave phantom of the sine → octave-suspect (strict rule).
        {"note_index": 4, "start": 6.0, "end": 7.0, "pitch": _SINE_PITCH + 12, "cluster_id": 3},
        # 5: a claim contradicting the audio (D#3 against the A3 sine) —
        # carried by the "wrong" section, which must band bad.
        {"note_index": 5, "start": 7.6, "end": 8.4, "pitch": 51, "cluster_id": 4},
        # 6: pitch underivable (no pitch, no string) → unscored, and EXCLUDED
        # from cross-model matching: agreement stays null, never 'mt3-only'.
        {"note_index": 6, "start": 8.8, "end": 8.9, "fret": 2, "cluster_id": 5},
    ]
    paths.frets_json.write_text(
        json.dumps(
            {
                "note_count": len(notes),
                "tuning": {"low_to_high_midi": [40, 45, 50, 55, 59, 64]},
                "notes": notes,
            }
        )
    )

    paths.sections_json.write_text(
        json.dumps(
            {
                "video_id": _VID,
                "sections": [
                    {
                        "label": "clean",
                        "instances": [
                            {"start": 0.0, "end": 5.0, "demo_quality": "slow-walkthrough"}
                        ],
                    },
                    {
                        "label": "wrong",
                        "instances": [
                            # render's preference: slow-walkthrough beats the
                            # earlier partial even though both share a window.
                            {"start": 7.4, "end": 8.6, "demo_quality": "partial"},
                            {"start": 7.4, "end": 8.6, "demo_quality": "slow-walkthrough"},
                        ],
                    },
                    {
                        "label": "empty",
                        "instances": [
                            {"start": 100.0, "end": 105.0, "demo_quality": "normal-tempo"}
                        ],
                    },
                ],
            }
        )
    )

    tuning_verification = {"checked": True, "veto": False, "subfloor_sustained_count": 0}
    paths.tuning_json.write_text(
        json.dumps(
            {
                "strings_midi": [40, 45, 50, 55, 59, 64],
                "capo": 0,
                "label": "Standard",
                "confidence": 1.0,
                "source": "audio",
                "evidence": "test fixture",
                "verification": tuning_verification,
            }
        )
    )

    mt3_notes = [
        {"start": n["start"], "end": n["end"], "pitch": n["pitch"], "velocity": 100}
        for n in notes
        if "pitch" in n  # note 6 has no pitch by design
    ]
    # One sustained sub-floor note (below Standard's low E2=40) for the
    # capo recompute to count.
    mt3_notes.append({"start": 8.0, "end": 8.3, "pitch": 38, "velocity": 100})
    paths.notes_mt3_json.write_text(json.dumps({"backend": "mt3", "notes": mt3_notes}))

    bp_notes = [
        {"start": 0.55, "end": 1.5, "pitch": 48, "velocity": 80},  # matches note 0 (+50 ms)
        {"start": 2.0, "end": 3.0, "pitch": 60, "velocity": 80},  # matches note 1
        {"start": 3.65, "end": 4.5, "pitch": 64, "velocity": 80},  # +150 ms → NO match for note 2
        {"start": 6.02, "end": 7.0, "pitch": 57, "velocity": 80},  # matches note 3
        {"start": 4.2, "end": 4.4, "pitch": 72, "velocity": 80},  # bp-only
    ]
    paths.notes_json.write_text(json.dumps({"notes": bp_notes}))
    return paths


class TestVerifyIntegration:
    @pytest.fixture(scope="class")
    def report_and_paths(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("verify")
        paths = _fixture_paths(tmp_path)
        return verify(paths), paths

    def test_verdicts(self, report_and_paths):
        report, _ = report_and_paths
        verdicts = {row[0]: row[2] for row in report["per_note"]}
        for i in (0, 1, 2, 3):
            assert verdicts[i] == VERDICT_SUPPORTED, f"note {i}"
        assert verdicts[4] == VERDICT_OCTAVE
        assert verdicts[6] == VERDICT_UNSCORED  # pitch underivable

    def test_agreement_tiers_and_counts(self, report_and_paths):
        report, _ = report_and_paths
        agreement = {row[0]: row[3] for row in report["per_note"]}
        assert agreement[0] == AGREEMENT_BOTH
        assert agreement[1] == AGREEMENT_BOTH
        assert agreement[2] == AGREEMENT_MT3_ONLY  # 150 ms > tolerance
        assert agreement[3] == AGREEMENT_BOTH
        # Underivable pitch → excluded from matching, NOT labeled mt3-only.
        assert agreement[6] is None
        assert report["summary"]["agreement"]["pitch_unknown"] == 1
        # bp-only = basic-pitch notes no tab note claims: the pitch-72 extra
        # AND the jittered 64 that fell outside the matching tolerance.
        assert report["summary"]["agreement"]["bp-only"] == 2

    def test_section_banding(self, report_and_paths):
        report, _ = report_and_paths
        bands = {s["label"]: s["band"] for s in report["per_section"]}
        assert bands["clean"] == BAND_SOLID  # stem == synth in that window
        assert bands["wrong"] == BAND_BAD
        assert bands["empty"] == BAND_NO_NOTES

    def test_canonical_instance_mirrors_render_preference(self, report_and_paths):
        report, _ = report_and_paths
        wrong = next(s for s in report["per_section"] if s["label"] == "wrong")
        assert wrong["demo_quality"] == "slow-walkthrough"

    def test_capo_passthrough_and_recompute(self, report_and_paths):
        report, _ = report_and_paths
        capo = report["summary"]["capo_check"]
        assert capo["present"] is True
        assert capo["tuning_verification"] == {
            "checked": True,
            "veto": False,
            "subfloor_sustained_count": 0,
        }
        recomputed = capo["recomputed"]
        assert recomputed["effective_low_midi"] == 40
        assert recomputed["subfloor_sustained_count"] == 1  # the MIDI-38 note
        assert recomputed["would_veto"] is False

    def test_schema(self, report_and_paths):
        report, paths = report_and_paths
        assert set(report) == {
            "video_id",
            "generated_at",
            "inputs",
            "per_note_columns",
            "per_note",
            "per_section",
            "summary",
        }
        assert report["per_note_columns"] == ["note_index", "salience", "verdict", "agreement"]
        assert len(report["per_note"]) == 7
        assert sum(report["summary"]["verdicts"].values()) == 7
        assert set(report["summary"]["verdicts"]) == set(VERDICTS)
        on_disk = json.loads(paths.verification_json.read_text())
        assert on_disk == report

    def test_generated_at_derives_from_input_mtimes(self, report_and_paths):
        report, paths = report_and_paths
        newest = max(
            p.stat().st_mtime
            for p in (
                paths.frets_json,
                paths.notes_mt3_json,
                paths.notes_json,
                paths.sections_json,
                paths.tuning_json,
                paths.guitar_stem,
            )
        )
        expected = datetime.fromtimestamp(newest, tz=UTC).isoformat()
        assert report["generated_at"] == expected

    def test_rerun_is_deterministic(self, report_and_paths):
        report, paths = report_and_paths
        assert verify(paths) == report

    def test_out_path_is_injectable(self, report_and_paths, tmp_path):
        _, paths = report_and_paths
        out = tmp_path / "elsewhere" / "verification.json"
        report = verify(paths, out_path=out)
        assert json.loads(out.read_text()) == report


class TestAgreementReasons:
    def test_empty_notes_json_reason_distinguished(self, tmp_path):
        """An existing-but-empty notes.json must not claim 'not available'."""
        paths = _fixture_paths(tmp_path)
        paths.notes_json.write_text(json.dumps({"notes": []}))
        report = verify(paths, out_path=tmp_path / "verification.json")
        ag = report["summary"]["agreement"]
        assert ag["checked"] is False
        assert "contains no usable notes" in ag["reason"]

    def test_missing_notes_json_reason(self, tmp_path):
        paths = _fixture_paths(tmp_path)
        paths.notes_json.unlink()
        report = verify(paths, out_path=tmp_path / "verification.json")
        ag = report["summary"]["agreement"]
        assert ag["checked"] is False
        assert "not available" in ag["reason"]


class TestVerifyErrors:
    def test_missing_frets_raises(self, tmp_path):
        paths = VideoPaths("testverify1", cache_dir=tmp_path)
        with pytest.raises(VerifyError, match="frets.json"):
            verify(paths)

    def test_missing_stem_raises(self, tmp_path):
        paths = VideoPaths("testverify2", cache_dir=tmp_path)
        paths.frets_json.write_text(json.dumps({"notes": []}))
        with pytest.raises(VerifyError, match="stem"):
            verify(paths)


# ---------------------------------------------------------------------------
# Real-cache checks (skipped on CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ANGIE_STEM.exists(), reason="local cache not available")
class TestAngieCache:
    def test_proven_phantom_b4_strums_flagged_octave_suspect(self):
        """The six B4=71 phantom strums in the open-E7 cluster (903.2-904.8 s)
        — the flagship wave-1 calibration case — must all flag."""
        from migs_tab.salience import compute_cqt_context

        frets = json.loads((_ANGIE_STEM.parents[1] / "frets.json").read_text())
        targets = [n for n in frets["notes"] if n["pitch"] == 71 and 903.1 <= n["start"] <= 904.9]
        assert len(targets) == 6
        y, sr = load_stem_window(_ANGIE_STEM, 901.0, 907.0)
        ctx = compute_cqt_context(y, sr, 901.0)
        for n in targets:
            _, verdict = assess_note(ctx, n["start"], n["pitch"])
            assert verdict == VERDICT_OCTAVE, f"note at {n['start']}"

    def test_proven_phantom_a5_onsets_flagged_octave_suspect(self):
        """The three A5=81 phantom onsets (828.4-829.1 s) ring at energy
        ratios INSIDE the genuine-octave-pair range, so only the LOOSE rule
        catches them — this pins the OCTAVE_LOOSE_* calibration on the real
        stem (the strict-rule B4 case above cannot cover it)."""
        from migs_tab.salience import compute_cqt_context

        frets = json.loads((_ANGIE_STEM.parents[1] / "frets.json").read_text())
        targets = [n for n in frets["notes"] if n["pitch"] == 81 and 828.3 <= n["start"] <= 829.2]
        assert len(targets) == 3
        y, sr = load_stem_window(_ANGIE_STEM, 826.0, 831.0)
        ctx = compute_cqt_context(y, sr, 826.0)
        for n in targets:
            _, verdict = assess_note(ctx, n["start"], n["pitch"])
            assert verdict == VERDICT_OCTAVE, f"note at {n['start']}"


@pytest.mark.skipif(not _LBTD_TUNING.exists(), reason="local cache not available")
class TestLbtdCache:
    def test_wrong_capo_recompute_would_veto(self):
        """The OLD broken LBTD artifacts claim 'Standard, capo 5'; the
        recompute against notes.mt3.json must contradict it — this is the
        only check that catches wrong-capo errors."""
        paths = VideoPaths("9jswOBilMvA", cache_dir=_CACHE_ROOT)
        out = capo_check(paths)
        assert out["capo"] == 5
        recomputed = out["recomputed"]
        assert recomputed["would_veto"] is True
        assert recomputed["subfloor_sustained_count"] >= 100
