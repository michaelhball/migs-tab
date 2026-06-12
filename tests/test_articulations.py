"""Tests for articulations.py — audio-gated hammer/pull/slide, bend and
natural-harmonic detection, plus the fret.py wiring contract.

Synthetic fixtures throughout: Karplus-Strong plucks carry the broadband
pick transient (their excitation is white noise), faded sines approximate
hammered/pulled notes (energy continues, no >2 kHz percussive burst), and
phase-integrated glides synthesize bends. One cache-gated test runs the
detector against the real Angie stem (skipped when cache/ is absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from migs_tab.articulations import (
    DEFAULT_WINDOW_SECONDS,
    _legato_type,
    detect_articulations,
)
from migs_tab.fret import STANDARD_TUNING, assign_frets
from migs_tab.paths import VideoPaths
from migs_tab.salience import DEFAULT_SR

_CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache" / "wS_i91qxQYM"

# A 10-note picked backdrop spread through the window gives the percentile
# gate a population (>= 8 cluster onsets required).
_BACKDROP_PITCHES = (45, 47, 48, 50, 52, 55, 57, 59, 60, 64)


def _ks_pluck(midi: int, dur: float, sr: int = DEFAULT_SR, seed: int = 3) -> np.ndarray:
    """Karplus-Strong pluck — sharp broadband attack (picked note)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    period = max(2, int(round(sr / librosa.midi_to_hz(midi))))
    buf = rng.uniform(-1.0, 1.0, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        if i % period == period - 1:
            buf = 0.996 * 0.5 * (buf + np.roll(buf, -1))
    out[-min(128, n) :] *= np.linspace(1, 0, min(128, n))
    return 0.6 * out


def _legato_tone(midi: int, dur: float, sr: int = DEFAULT_SR) -> np.ndarray:
    """Faded sine — energy with no percussive attack (hammered/pulled)."""
    t = np.arange(int(dur * sr)) / sr
    y = 0.5 * np.sin(2 * np.pi * librosa.midi_to_hz(midi) * t) * np.exp(-1.5 * t)
    ramp = min(int(0.015 * sr), len(y))
    y[:ramp] *= np.linspace(0, 1, ramp)
    y[-ramp:] *= np.linspace(1, 0, ramp)
    return y


def _glide_tone(
    midi_from: int, midi_to: int, hold: float, glide: float, sustain: float, sr: int = DEFAULT_SR
) -> np.ndarray:
    """Phase-integrated pitch glide: hold midi_from, glide, sustain midi_to."""
    n_hold, n_glide, n_sus = (int(x * sr) for x in (hold, glide, sustain))
    midi_curve = np.concatenate(
        [
            np.full(n_hold, float(midi_from)),
            np.linspace(midi_from, midi_to, max(n_glide, 1)),
            np.full(n_sus, float(midi_to)),
        ]
    )
    freq = librosa.midi_to_hz(midi_curve)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    y = 0.5 * np.sin(phase) * np.exp(-0.8 * np.arange(len(phase)) / sr)
    ramp = min(int(0.015 * sr), len(y))
    y[:ramp] *= np.linspace(0, 1, ramp)
    y[-ramp:] *= np.linspace(1, 0, ramp)
    return y


def _mix(events: list[tuple[float, np.ndarray]], total: float, sr: int = DEFAULT_SR) -> np.ndarray:
    out = np.zeros(int(total * sr))
    for onset, y in events:
        i0 = int(onset * sr)
        out[i0 : i0 + len(y)] += y[: max(0, len(out) - i0)]
    return out


def _write_stem(tmp_path: Path, y: np.ndarray, sr: int = DEFAULT_SR) -> Path:
    stem = tmp_path / "stem.wav"
    sf.write(stem, y.astype(np.float32), sr)
    return stem


def _record(i: int, start: float, end: float, pitch: int, string: int, fret: int, cluster: int):
    return {
        "note_index": i,
        "start": start,
        "end": end,
        "pitch": pitch,
        "string": string,
        "fret": fret,
        "cluster_id": cluster,
    }


def _backdrop(start_index: int, start_cluster: int) -> tuple[list, list[tuple[float, np.ndarray]]]:
    """Ten picked notes at t = 4.0, 4.8, ... 11.2 s on the low E string."""
    records, events = [], []
    for k, pitch in enumerate(_BACKDROP_PITCHES):
        t = 4.0 + 0.8 * k
        records.append(
            _record(start_index + k, t, t + 0.5, pitch, 0, pitch - 40, start_cluster + k)
        )
        events.append((t, _ks_pluck(pitch, 0.5, seed=10 + k)))
    return records, events


class TestLegatoTyping:
    """_legato_type is the pure candidate classifier."""

    def test_fretted_small_deltas(self):
        assert _legato_type(1, 2) == "hammer"
        assert _legato_type(2, 1) == "pull"
        assert _legato_type(2, 4) == "hammer"

    def test_open_endpoint_pairs(self):
        # Can't slide from/to an open string: 0h3 and 3p0 are legato.
        assert _legato_type(0, 3) == "hammer"
        assert _legato_type(3, 0) == "pull"
        assert _legato_type(0, 4) == "hammer"
        # ...but wider open-endpoint moves are position changes, not slides.
        assert _legato_type(0, 7) is None
        assert _legato_type(7, 0) is None

    def test_fretted_slides_and_limits(self):
        assert _legato_type(2, 7) == "slide"
        assert _legato_type(9, 3) == "slide"
        assert _legato_type(2, 2) is None
        assert _legato_type(2, 12) is None  # > _SLIDE_MAX_DELTA


class TestHammerPullSlide:
    def _run(self, tmp_path, second_audio, from_fret=2, to_fret=4, string=2, backdrop=True):
        """Picked first note (D string fret ``from_fret``) then ``second_audio``
        at ``to_fret`` 0.3 s later, over the picked backdrop."""
        open_pitch = STANDARD_TUNING[string]
        records, events = _backdrop(100, 100) if backdrop else ([], [])
        records += [
            _record(0, 2.0, 2.28, open_pitch + from_fret, string, from_fret, 0),
            _record(1, 2.3, 2.9, open_pitch + to_fret, string, to_fret, 1),
        ]
        events += [(2.0, _ks_pluck(open_pitch + from_fret, 0.3)), (2.3, second_audio)]
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        return detect_articulations(records, stem, STANDARD_TUNING)

    def test_hammer_detected_for_legato_second_note(self, tmp_path):
        arts = self._run(tmp_path, _legato_tone(STANDARD_TUNING[2] + 4, 0.6))
        hammers = [a for a in arts if a["type"] == "hammer"]
        assert len(hammers) == 1
        (h,) = hammers
        assert h["from_note_index"] == 0
        assert h["note_index"] == 1
        assert h["string"] == 2
        assert (h["from_fret"], h["to_fret"]) == (2, 4)
        ev = h["evidence"]
        assert ev["onset_ratio"] < 0.9
        assert ev["attack_pctl"] <= 0.25

    def test_pull_detected_for_descending_pair(self, tmp_path):
        arts = self._run(
            tmp_path, _legato_tone(STANDARD_TUNING[2] + 2, 0.6), from_fret=4, to_fret=2
        )
        assert [a["type"] for a in arts] == ["pull"]

    def test_slide_detected_for_wide_fretted_delta(self, tmp_path):
        arts = self._run(
            tmp_path, _legato_tone(STANDARD_TUNING[2] + 7, 0.6), from_fret=2, to_fret=7
        )
        assert [a["type"] for a in arts] == ["slide"]

    def test_picked_second_note_not_marked(self, tmp_path):
        # The audio gate is the whole point: same intervals, same timing,
        # but a picked (sharp-attack) second note must yield NO mark.
        arts = self._run(tmp_path, _ks_pluck(STANDARD_TUNING[2] + 4, 0.6, seed=4))
        assert arts == []

    def test_sparse_window_fails_closed(self, tmp_path):
        # Same legato audio as the detected hammer above, but WITHOUT the
        # picked backdrop the window holds only 2 cluster onsets — below
        # _MIN_PCTL_POPULATION the percentile gate must fail closed (no
        # detection), never fall back to the absolute cuts alone.
        arts = self._run(tmp_path, _legato_tone(STANDARD_TUNING[2] + 4, 0.6), backdrop=False)
        assert arts == []

    def test_target_inside_strum_cluster_not_marked(self, tmp_path):
        # Legato second note whose cluster holds 3 notes (a strum): the
        # soft attack describes the strum, not a one-finger action.
        open_pitch = STANDARD_TUNING[2]
        records, events = _backdrop(100, 100)
        records += [
            _record(0, 2.0, 2.28, open_pitch + 2, 2, 2, 0),
            _record(1, 2.3, 2.9, open_pitch + 4, 2, 4, 1),
            _record(2, 2.31, 2.9, STANDARD_TUNING[3] + 2, 3, 2, 1),
            _record(3, 2.32, 2.9, STANDARD_TUNING[4] + 1, 4, 1, 1),
        ]
        events += [(2.0, _ks_pluck(open_pitch + 2, 0.3)), (2.3, _legato_tone(open_pitch + 4, 0.6))]
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        assert [a for a in arts if a["type"] in ("hammer", "pull", "slide")] == []


class TestBend:
    def _bend_fixture(self, tmp_path, struck_fret=3, glide=True, backdrop=True):
        """D-string fret 3 struck at 2.0 s, pitch glides +2 from 2.7 s, and
        the transcriber 'caught' the arrival as a member note at 2.9 s."""
        string = 2
        p0 = STANDARD_TUNING[string] + struck_fret
        records, events = _backdrop(100, 100) if backdrop else ([], [])
        records += [
            _record(0, 2.0, 2.85, p0, string, struck_fret, 0),
            _record(1, 2.9, 3.6, p0 + 2, string, struck_fret + 2, 1),
        ]
        if glide:
            events.append((2.0, _glide_tone(p0, p0 + 2, hold=0.7, glide=0.18, sustain=0.8)))
        else:
            events.append((2.0, _legato_tone(p0, 1.7)))
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        return records, stem

    def test_bend_detected_with_glide_and_member_hidden(self, tmp_path):
        records, stem = self._bend_fixture(tmp_path)
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        bends = [a for a in arts if a["type"] == "bend"]
        assert len(bends) == 1
        (b,) = bends
        assert b["note_index"] == 0
        assert b["string"] == 2
        assert b["fret"] == 3
        assert b["target_semitones"] == 2
        assert b["member_note_indices"] == [1]
        assert b["evidence"]["tail_pitch_track"] == pytest.approx(STANDARD_TUNING[2] + 5, abs=0.35)
        # The member is spoken for — it must not double as a hammer target.
        assert [a for a in arts if a["type"] == "hammer"] == []

    def test_no_bend_without_pitch_migration(self, tmp_path):
        # Source keeps ringing at its own pitch — the member is phantom.
        records, stem = self._bend_fixture(tmp_path, glide=False)
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        assert [a for a in arts if a["type"] == "bend"] == []

    def test_open_string_never_a_bend(self, tmp_path):
        # Same glide audio, but the struck note sits at fret 0 — an open
        # string cannot be bent, whatever the track says.
        records, stem = self._bend_fixture(tmp_path)
        records_open = []
        for r in records:
            r = dict(r)
            if r["note_index"] in (0, 1):
                r["fret"] -= 3
            records_open.append(r)
        arts = detect_articulations(records_open, stem, STANDARD_TUNING)
        assert [a for a in arts if a["type"] == "bend"] == []

    def test_member_outside_sustain_not_a_candidate(self, tmp_path):
        records, stem = self._bend_fixture(tmp_path)
        late = []
        for r in records:
            r = dict(r)
            if r["note_index"] == 1:
                r["start"], r["end"] = 3.2, 3.9  # past struck end (2.85) + tol
            late.append(r)
        arts = detect_articulations(late, stem, STANDARD_TUNING)
        assert [a for a in arts if a["type"] == "bend"] == []

    def test_sparse_window_fails_closed(self, tmp_path):
        # Real glide audio, but without the backdrop population the
        # member-attack percentile gate must fail closed: no bend (and no
        # anything) out of a window with < _MIN_PCTL_POPULATION onsets.
        records, stem = self._bend_fixture(tmp_path, backdrop=False)
        assert detect_articulations(records, stem, STANDARD_TUNING) == []

    def _boundary_fixture(self, tmp_path, gap: float):
        """Picked strike at t = 0 whose pitch glides +2 semitones, landing
        exactly at the member's onset ``gap`` seconds later. t = 0 keeps the
        computed inter-onset gap float-exact (gap - 0.0 == gap), so the
        detector sees PRECISELY the boundary value."""
        string = 2
        p0 = STANDARD_TUNING[string] + 3
        records, events = _backdrop(100, 100)
        records += [
            _record(0, 0.0, gap - 0.03, p0, string, 3, 0),
            _record(1, gap, gap + 0.7, p0 + 2, string, 5, 1),
        ]
        events += [
            (0.0, _ks_pluck(p0, 0.2)),
            (0.0, _glide_tone(p0, p0 + 2, hold=gap - 0.12, glide=0.12, sustain=0.9)),
        ]
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        return detect_articulations(records, stem, STANDARD_TUNING)

    def test_gap_at_legato_boundary_routes_to_hammer(self, tmp_path):
        # gap == _MAX_LEGATO_GAP exactly: the boundary belongs to the LEGATO
        # side. The same glide audio that yields a bend just past the
        # boundary must come out as a single hammer here — never both marks,
        # never a bend.
        arts = self._boundary_fixture(tmp_path, 0.45)
        assert [a["type"] for a in arts] == ["hammer"]

    def test_gap_just_past_boundary_stays_a_bend(self, tmp_path):
        arts = self._boundary_fixture(tmp_path, 0.5)
        bends = [a for a in arts if a["type"] == "bend"]
        assert len(bends) == 1
        assert bends[0]["member_note_indices"] == [1]
        assert [a for a in arts if a["type"] in ("hammer", "pull", "slide")] == []


class TestHarmonic:
    def _harmonic_tone(self, midi: int, dur: float, partials: dict[int, float]) -> np.ndarray:
        sr = DEFAULT_SR
        t = np.arange(int(dur * sr)) / sr
        y = 0.5 * np.sin(2 * np.pi * librosa.midi_to_hz(midi) * t)
        for offset, amp in partials.items():
            y += amp * np.sin(2 * np.pi * librosa.midi_to_hz(midi + offset) * t)
        y *= np.exp(-0.5 * t)
        ramp = min(int(0.01 * sr), len(y))
        y[:ramp] *= np.linspace(0, 1, ramp)
        y[-ramp:] *= np.linspace(1, 0, ramp)
        return y

    def test_pure_long_isolated_node_pitch_flagged(self, tmp_path):
        # A3 = open A (45) + 12, nearly pure, 1.5 s, alone in its cluster:
        # the Angie t=1.0 s pickup pattern.
        y = self._harmonic_tone(57, 1.5, {12: 0.04, 19: 0.02})
        stem = _write_stem(tmp_path, _mix([(1.0, y)], 4.0))
        records = [_record(0, 1.0, 2.5, 57, 3, 2, 0)]
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        harmonics = [a for a in arts if a["type"] == "harmonic"]
        assert len(harmonics) == 1
        (h,) = harmonics
        assert h["note_index"] == 0
        assert h["open_string"] == 1  # re-strung from G|2 to the A string
        assert h["node_fret"] == 12
        assert h["evidence"]["purity"] >= 0.25

    def test_fretted_partial_profile_rejected(self, tmp_path):
        # Same pitch/duration but with a fretted note's overtones (strong
        # +12/+19) and the open-A fundamental ringing below — all three
        # spectral cuts the LBTD fretted-A3 negatives failed.
        y = self._harmonic_tone(57, 1.5, {12: 0.35, 19: 0.25, -12: 0.2})
        stem = _write_stem(tmp_path, _mix([(1.0, y)], 4.0))
        records = [_record(0, 1.0, 2.5, 57, 3, 2, 0)]
        assert detect_articulations(records, stem, STANDARD_TUNING) == []

    def test_short_or_clustered_notes_skipped(self, tmp_path):
        y = self._harmonic_tone(57, 1.5, {12: 0.04})
        stem = _write_stem(tmp_path, _mix([(1.0, y)], 4.0))
        # Too short.
        records = [_record(0, 1.0, 1.5, 57, 3, 2, 0)]
        assert detect_articulations(records, stem, STANDARD_TUNING) == []
        # Long enough but sharing a cluster (not isolated).
        records = [
            _record(0, 1.0, 2.5, 57, 3, 2, 0),
            _record(1, 1.02, 1.4, 64, 5, 0, 0),
        ]
        assert detect_articulations(records, stem, STANDARD_TUNING) == []

    def test_non_node_pitch_skipped(self, tmp_path):
        # 58 = no open string + 12 under Standard tuning.
        y = self._harmonic_tone(58, 1.5, {12: 0.04})
        stem = _write_stem(tmp_path, _mix([(1.0, y)], 4.0))
        records = [_record(0, 1.0, 2.5, 58, 3, 3, 0)]
        assert detect_articulations(records, stem, STANDARD_TUNING) == []

    def test_drop_d_open_string_ordinary_partials_rejected(self, tmp_path):
        # Tuning-specific aliasing: under Drop D the open D string (50)
        # IS the low string's 12th-fret node pitch (38 + 12), so every long
        # isolated open D becomes a harmonic candidate. An ordinary open
        # string's strong +12/+19 partials must fail the spectral cuts —
        # all partials (62, 69, 38) sit inside the salience CQT range
        # (MIDI 36-93), so this exercises the cuts, not the range guard.
        drop_d = (38, 45, 50, 55, 59, 64)
        y = self._harmonic_tone(50, 1.5, {12: 0.35, 19: 0.25})
        stem = _write_stem(tmp_path, _mix([(1.0, y)], 4.0))
        records = [_record(0, 1.0, 2.5, 50, 2, 0, 0)]
        assert detect_articulations(records, stem, drop_d) == []


class TestCrossDetectorDisjointness:
    """No entry may reference a note that another entry orders hidden or
    re-strung: bends never chain through their own members, and a note the
    harmonic detector claims is never a bend endpoint."""

    def test_chained_bends_never_share_notes(self, tmp_path):
        # Bend-release-rebend lick: the pitch glides p0 -> p0+2 -> p0+4 and
        # the transcriber caught all three plateaus, so the candidate pairs
        # chain (a,b) then (b,c). b is the FIRST bend's hidden arrival, not
        # a new strike — only (a,b) may survive.
        sr = DEFAULT_SR
        string = 2
        p0 = STANDARD_TUNING[string] + 3
        curve = np.concatenate(
            [
                np.full(int(0.7 * sr), float(p0)),
                np.linspace(p0, p0 + 2, int(0.18 * sr)),
                np.full(int(0.82 * sr), float(p0 + 2)),
                np.linspace(p0 + 2, p0 + 4, int(0.18 * sr)),
                np.full(int(0.8 * sr), float(p0 + 4)),
            ]
        )
        freq = librosa.midi_to_hz(curve)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        y = 0.5 * np.sin(phase) * np.exp(-0.6 * np.arange(len(phase)) / sr)
        ramp = int(0.015 * sr)
        y[:ramp] *= np.linspace(0, 1, ramp)
        y[-ramp:] *= np.linspace(1, 0, ramp)
        records, events = _backdrop(100, 100)
        records += [
            _record(0, 2.0, 2.85, p0, string, 3, 0),
            _record(1, 2.9, 3.6, p0 + 2, string, 5, 1),
            _record(2, 3.8, 4.6, p0 + 4, string, 7, 2),
        ]
        events.append((2.0, y))
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        bends = [a for a in arts if a["type"] == "bend"]
        assert len(bends) == 1
        assert bends[0]["note_index"] == 0
        assert bends[0]["member_note_indices"] == [1]
        # The re-bend arrival (note 2) gets no mark of any kind.
        assert all(a["note_index"] != 2 for a in arts)

    def test_harmonic_note_not_a_bend_endpoint(self, tmp_path):
        # A long pure A3 (open A + 12) that passes the harmonic gates, then
        # a glide up to B3 caught as a separate note: without exclusion the
        # bend detector would mark the SAME note the harmonic entry orders
        # re-strung. The harmonic must win and no bend may appear.
        records, events = _backdrop(100, 100)
        records += [
            _record(0, 1.0, 1.85, 57, 3, 2, 0),
            _record(1, 1.9, 2.6, 59, 3, 4, 1),
        ]
        events.append((1.0, _glide_tone(57, 59, hold=0.85, glide=0.1, sustain=0.7)))
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        assert [a["type"] for a in arts] == ["harmonic"]
        assert arts[0]["note_index"] == 0


class TestContractShape:
    def test_empty_records_empty_result(self, tmp_path):
        stem = _write_stem(tmp_path, np.zeros(DEFAULT_SR))
        assert detect_articulations([], stem, STANDARD_TUNING) == []

    def test_entries_match_contract_fields(self, tmp_path):
        string = 2
        p0 = STANDARD_TUNING[string]
        records, events = _backdrop(100, 100)
        records += [
            _record(0, 2.0, 2.28, p0 + 2, string, 2, 0),
            _record(1, 2.3, 2.9, p0 + 4, string, 4, 1),
        ]
        events += [(2.0, _ks_pluck(p0 + 2, 0.3)), (2.3, _legato_tone(p0 + 4, 0.6))]
        stem = _write_stem(tmp_path, _mix(events, 13.0))
        arts = detect_articulations(records, stem, STANDARD_TUNING)
        assert arts, "fixture must produce at least one articulation"
        for a in arts:
            assert a["type"] in ("hammer", "pull", "slide", "bend", "harmonic")
            assert isinstance(a["note_index"], int)
            assert isinstance(a["evidence"], dict)
            if a["type"] in ("hammer", "pull", "slide"):
                assert set(a) == {
                    "type",
                    "from_note_index",
                    "note_index",
                    "string",
                    "from_fret",
                    "to_fret",
                    "evidence",
                }
                assert isinstance(a["evidence"]["onset_ratio"], float)
            elif a["type"] == "bend":
                assert set(a) == {
                    "type",
                    "note_index",
                    "string",
                    "fret",
                    "target_semitones",
                    "member_note_indices",
                    "evidence",
                }
                assert a["target_semitones"] in (1, 2)
            else:
                assert set(a) == {"type", "note_index", "open_string", "node_fret", "evidence"}


class TestFretWiring:
    """assign_frets embeds articulations only when a stem exists AND the
    detector returns something."""

    def _write_notes(self, paths: VideoPaths, notes: list[dict]) -> None:
        paths.notes_mt3_json.write_text(json.dumps({"notes": notes}))

    def _write_sine_stem(self, paths: VideoPaths) -> None:
        sr = DEFAULT_SR
        t = np.arange(int(2.0 * sr)) / sr
        y = 0.5 * np.sin(2 * np.pi * 220.0 * t)
        paths.stems_dir.mkdir(parents=True, exist_ok=True)
        sf.write(paths.guitar_stem, y.astype(np.float32), sr)

    def test_no_stem_no_articulations_key(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_notes(paths, [{"start": 0.5, "end": 1.0, "pitch": 57, "velocity": 100}])
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert out["params"]["audio_evidence"] is False
        assert "articulations" not in out

    def test_stem_with_no_detections_omits_key(self, tmp_path, monkeypatch):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_sine_stem(paths)
        self._write_notes(paths, [{"start": 0.5, "end": 1.0, "pitch": 57, "velocity": 100}])
        monkeypatch.setattr("migs_tab.articulations.detect_articulations", lambda *a, **k: [])
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert out["params"]["audio_evidence"] is True
        assert "articulations" not in out

    def test_stem_with_detections_embeds_list(self, tmp_path, monkeypatch):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        self._write_sine_stem(paths)
        self._write_notes(paths, [{"start": 0.5, "end": 1.0, "pitch": 57, "velocity": 100}])
        sentinel = [
            {
                "type": "harmonic",
                "note_index": 0,
                "open_string": 1,
                "node_fret": 12,
                "evidence": {"purity": 0.31},
            }
        ]
        captured: dict = {}

        def fake(note_records, stem_path, tuning, contexts=None, window_seconds=None):
            captured["records"] = note_records
            captured["stem_path"] = Path(stem_path)
            captured["tuning"] = tuple(tuning)
            captured["contexts"] = contexts
            captured["window_seconds"] = window_seconds
            return sentinel

        monkeypatch.setattr("migs_tab.articulations.detect_articulations", fake)
        assign_frets(paths, force=True)
        out = json.loads(paths.frets_json.read_text())
        assert out["articulations"] == sentinel
        # The wiring hands over the final note records, the stem path, the
        # active tuning and the SHARED evidence contexts (so the CQT isn't
        # recomputed) keyed by the same window size.
        assert captured["records"] == out["notes"]
        assert captured["stem_path"] == paths.guitar_stem
        assert captured["tuning"] == tuple(out["tuning"]["low_to_high_midi"])
        assert captured["contexts"] is not None
        assert captured["window_seconds"] == DEFAULT_WINDOW_SECONDS


@pytest.mark.skipif(not _CACHE_ROOT.exists(), reason="local Angie cache not present")
class TestRealStemRegression:
    """The labeled Angie t=1.0 s natural harmonic, detected from the real
    stem (window 0 only — records are sliced to keep the test fast)."""

    def test_flagship_harmonic_detected(self):
        frets = json.loads((_CACHE_ROOT / "frets.json").read_text())
        records = [n for n in frets["notes"] if n["start"] < 79.0]
        arts = detect_articulations(
            records,
            _CACHE_ROOT / "stems" / "other.wav",
            tuple(frets["tuning"]["low_to_high_midi"]),
        )
        harmonics = [a for a in arts if a["type"] == "harmonic"]
        assert len(harmonics) == 1
        (h,) = harmonics
        assert h["note_index"] == 0
        assert h["open_string"] == 1
        assert h["node_fret"] == 12
        assert h["evidence"]["purity"] >= 0.25
