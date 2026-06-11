"""Verification pass: score the finished tab against the evidence.

``migs-tab verify`` answers "should I trust this tab?" AFTER the pipeline has
run, by combining every independent evidence source we have:

1. Per-note salience verdicts — for every note in frets.json, the percentile
   rank of the claimed pitch's CQT bin in the 30-200 ms post-onset window of
   the Demucs guitar stem (salience.salience_at), computed over one shared
   CQT per chunk of audio. Each note gets a verdict tier plus an octave check
   that flags notes whose ±12-semitone partner explains the audio better.
2. Cross-model agreement — greedy nearest-onset matching (±100 ms, exact
   pitch) between the YourMT3 and basic-pitch transcriptions. A note both
   models heard is far more trustworthy than a single-model claim.
3. Tuning/capo contradiction report — echoes tuning.json's wave-1
   ``verification`` block and RECOMPUTES the sub-floor evidence against the
   current notes files. This is the only check that catches wrong-capo
   errors: audio-only checks are blind to them because sounding pitches stay
   self-consistent under a wrong capo.
4. Section scores — per section (canonical instance, mirroring render.py's
   preference), salience.section_score Karplus-Strong-synthesizes the
   claimed notes and compares CQT chroma against the real stem at absolute
   timestamps (no DTW, by design — see salience.py).

Output: cache/<id>/verification.json (path injectable for tests/experiments)
plus a per-section table printed by the CLI. Bad scores never exit nonzero —
only hard errors (missing inputs) do; the report is advice, not a gate.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from datetime import UTC, datetime
from pathlib import Path

from . import salience
from .paths import VideoPaths

# READ-ONLY mirrors of behavior owned by other modules. Imported rather than
# reimplemented so verify can never drift from the pipeline it verifies:
# render's canonical-instance preference (slow-walkthrough > normal-tempo >
# longest) and tuning's calibrated sub-floor contradiction math.
from .render import _pick_canonical_instance
from .tuning import _subfloor_evidence, load_transcribed_notes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-note salience verdict tiers, applied to salience.salience_at's
# percentile rank (1.0 = the claimed pitch is the loudest thing ringing).
# Calibrated on cache/wS_i91qxQYM (Angie) frets.json notes:
#   chroma-verified-good windows (1107.2-1173.1 s slow replay + 170.4-184.4 s
#   picked walkthrough, 235 scored notes): p5 0.828, p10 0.862, p25 0.94,
#   median 0.983 — so >= 0.85 captures ~95 % of known-good notes.
#   Proven phantoms in the open-E7 cluster 903.2-904.8 s: G#4=68 scored
#   0.086-0.172 (all six strums below 0.60), B4=71 scored 0.362-0.931 (the
#   louder ones are only caught by the octave check below).
SALIENCE_SUPPORTED_MIN = 0.85
SALIENCE_WEAK_MIN = 0.60

VERDICT_SUPPORTED = "supported"
VERDICT_WEAK = "weak"
VERDICT_PHANTOM = "phantom-suspect"
VERDICT_OCTAVE = "octave-suspect"
VERDICT_UNSCORED = "unscored"
VERDICTS = (VERDICT_SUPPORTED, VERDICT_WEAK, VERDICT_PHANTOM, VERDICT_OCTAVE, VERDICT_UNSCORED)

# Octave check: a note p is 'octave-suspect' when an octave partner (p ± 12)
# explains the post-onset audio better. Two rules, both measured on Angie:
#
# STRICT (either direction): salience(p±12) >= salience(p) AND the energy
# ratio E(p)/E(p±12) < salience.OCTAVE_ARTIFACT_RATIO (0.20 — calibrated in
# salience.py on proven phantoms vs 28 genuine octave pairs). Catches 6/6
# B4=71 phantom strums at 903.2-904.8 s (ratios 0.017-0.123) with 1/235
# (0.4 %) false tags on the known-good windows.
#
# LOOSE (overtone direction p-12 only): the A5=81 phantom at 828.4-829.1 s
# rings at ratios 0.33-0.55 — inside the genuine-octave-pair range, so the
# strict ratio alone cannot catch it (salience.py documents this overlap).
# But its fundamental-down partner is near-top-ranked while the phantom
# trails it: require salience(p-12) >= _OCTAVE_LOOSE_ALT_MIN, a salience
# margin >= _OCTAVE_LOOSE_MARGIN, and ratio < _OCTAVE_LOOSE_RATIO. Catches
# 3/3 A5 phantom onsets with 4/235 (1.7 %) false tags on the known-good
# windows. The verdict is advisory (nothing is deleted on it), so a small
# false-positive rate is the right trade.
OCTAVE_STRICT_RATIO = salience.OCTAVE_ARTIFACT_RATIO
OCTAVE_LOOSE_RATIO = 0.60
OCTAVE_LOOSE_MARGIN = 0.04
OCTAVE_LOOSE_ALT_MIN = 0.97

# Cross-model agreement: onsets within this tolerance and EXACT pitch match.
AGREEMENT_ONSET_TOL_S = 0.100

AGREEMENT_BOTH = "both-models"
AGREEMENT_MT3_ONLY = "mt3-only"
AGREEMENT_BP_ONLY = "bp-only"

# Section-score bands. Calibrated on ONE Angie window in the salience.py
# spike (real tab 0.893, 15 % corrupted 0.843, random pitches 0.586) and
# confirmed on cached sections: Angie intro_full_slow_replay (chroma-verified
# good) scores 0.891 → solid; LBTD's known-bad [power_chord_A] 0.604 → bad,
# [Fsharpm_chord] 0.760 / [intro_triplet_feel] 0.784 → suspect. The floor of
# the metric sits around 0.59-0.61, so "bad" is close to chance agreement.
SECTION_BAND_SOLID_MIN = 0.85
SECTION_BAND_BAD_MAX = 0.70

BAND_SOLID = "solid"
BAND_SUSPECT = "suspect"
BAND_BAD = "bad"
BAND_NO_NOTES = "no-notes"

# Per-note CQT chunking: one shared CQT per chunk (the CQT is the dominant
# cost). Chunks are capped in span so memory stays bounded, and split at
# silent gaps so talking-only stretches of the video are never transformed.
_CHUNK_MAX_SPAN_S = 60.0
_CHUNK_GAP_SPLIT_S = 10.0
# Audio loaded past the last onset of a chunk (covers POST_ONSET_END plus
# CQT edge frames) and before the first one (CQT filter warm-up).
_CHUNK_PAD_S = 0.5
_CHUNK_PRE_ROLL_S = 0.1


class VerifyError(RuntimeError):
    """Verification could not run at all (missing/unreadable inputs)."""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _load_notes_list(path: Path) -> list[tuple[float, int]] | None:
    """(onset, pitch) events from a transcription JSON ({"notes": [...]} or a
    bare list), skipping malformed records. None when the file is missing or
    unusable; an EMPTY list when the file exists but holds no usable notes —
    callers report the two cases differently."""
    data = _load_json(path)
    raw = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return None
    events: list[tuple[float, int]] = []
    for n in raw:
        if not isinstance(n, dict):
            continue
        try:
            events.append((float(n["start"]), int(n["pitch"])))
        except (KeyError, TypeError, ValueError):
            continue
    return events


def _note_pitch(note: dict, tuning_midis: list[int] | None) -> int | None:
    """MIDI pitch of a frets.json note — the ``pitch`` field when present,
    otherwise derived from tuning + string + fret (frets.json embeds the
    effective tuning, capo already folded in)."""
    if "pitch" in note:
        try:
            return int(note["pitch"])
        except (TypeError, ValueError):
            return None
    if tuning_midis is None:
        return None
    try:
        string = int(note["string"])
        fret = int(note["fret"])
    except (KeyError, TypeError, ValueError):
        return None
    # Explicit bound check: a negative index would silently wrap (Python
    # indexing) and derive a wrong pitch instead of None.
    if not 0 <= string < len(tuning_midis):
        return None
    try:
        return int(tuning_midis[string]) + fret
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-note salience verdicts
# ---------------------------------------------------------------------------


def _tier_for_salience(s: float) -> str:
    if s >= SALIENCE_SUPPORTED_MIN:
        return VERDICT_SUPPORTED
    if s >= SALIENCE_WEAK_MIN:
        return VERDICT_WEAK
    return VERDICT_PHANTOM


def assess_note(ctx: salience.CQTContext, onset: float, pitch: int) -> tuple[float | None, str]:
    """(salience, verdict) for one claimed note against a prebuilt CQT.

    Computes the post-onset column once and derives the note's own percentile
    rank plus the ±12-semitone octave checks from it (see the octave-rule
    constants above for the calibration story).
    """
    b = ctx.bin_for_pitch(pitch)
    col = ctx.post_onset_column(onset) if b is not None else None
    if b is None or col is None or float(col[b]) <= 0.0:
        return None, VERDICT_UNSCORED
    e = float(col[b])
    s = float((col <= col[b]).mean())
    for delta in (-12, 12):
        ba = ctx.bin_for_pitch(pitch + delta)
        if ba is None or float(col[ba]) <= 0.0:
            continue
        s_alt = float((col <= col[ba]).mean())
        ratio = e / float(col[ba])
        if s_alt >= s and ratio < OCTAVE_STRICT_RATIO:
            return s, VERDICT_OCTAVE
        if (
            delta == -12
            and s_alt >= OCTAVE_LOOSE_ALT_MIN
            and s_alt >= s + OCTAVE_LOOSE_MARGIN
            and ratio < OCTAVE_LOOSE_RATIO
        ):
            return s, VERDICT_OCTAVE
    return s, _tier_for_salience(s)


def _chunk_indices(onsets: list[float]) -> list[list[int]]:
    """Group note indices into CQT chunks: bounded span, split at silences."""
    order = sorted(range(len(onsets)), key=lambda i: onsets[i])
    chunks: list[list[int]] = []
    current: list[int] = []
    chunk_start = 0.0
    prev_onset = 0.0
    for i in order:
        t = onsets[i]
        if current and (t - chunk_start > _CHUNK_MAX_SPAN_S or t - prev_onset > _CHUNK_GAP_SPLIT_S):
            chunks.append(current)
            current = []
        if not current:
            chunk_start = t
        current.append(i)
        prev_onset = t
    if current:
        chunks.append(current)
    return chunks


def assess_all_notes(
    notes: list[dict],
    pitches: list[int | None],
    stem_path: Path,
) -> list[tuple[float | None, str]]:
    """Salience + verdict per frets.json note, one shared CQT per chunk."""
    results: list[tuple[float | None, str]] = [(None, VERDICT_UNSCORED)] * len(notes)
    scorable = [i for i in range(len(notes)) if pitches[i] is not None]
    onsets = [float(n["start"]) for n in notes]
    for chunk in _chunk_indices([onsets[i] for i in scorable]):
        idx = [scorable[j] for j in chunk]
        start = max(0.0, onsets[idx[0]] - _CHUNK_PRE_ROLL_S)
        end = onsets[idx[-1]] + _CHUNK_PAD_S
        y, sr = salience.load_stem_window(stem_path, start, end)
        ctx = salience.compute_cqt_context(y, sr, start)
        for i in idx:
            pitch = pitches[i]
            assert pitch is not None  # filtered above
            results[i] = assess_note(ctx, onsets[i], pitch)
    return results


# ---------------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------------


def match_events(
    primary: list[tuple[float, int]],
    secondary: list[tuple[float, int]],
    tol: float = AGREEMENT_ONSET_TOL_S,
) -> tuple[list[bool], int]:
    """Greedy nearest-onset matching at exact pitch.

    Each secondary event is consumed by at most one primary event; primaries
    are processed in onset order and take the nearest unused secondary onset
    within ``tol`` (inclusive on both sides; equidistant ties prefer the
    earlier onset). Returns (matched flag per primary event, count of
    secondary events no primary matched).
    """
    by_pitch: dict[int, list[float]] = {}
    for onset, pitch in secondary:
        by_pitch.setdefault(pitch, []).append(onset)
    used: dict[int, list[bool]] = {}
    for pitch, lst in by_pitch.items():
        lst.sort()
        used[pitch] = [False] * len(lst)

    flags = [False] * len(primary)
    matched = 0
    for i in sorted(range(len(primary)), key=lambda i: primary[i][0]):
        onset, pitch = primary[i]
        cand = by_pitch.get(pitch)
        if not cand:
            continue
        flag = used[pitch]
        j = bisect_left(cand, onset)
        best = -1
        best_d = float("inf")  # loop bounds enforce <= tol; both sides agree
        k = j - 1
        while k >= 0 and onset - cand[k] <= tol:
            d = onset - cand[k]
            if not flag[k] and d < best_d:
                best, best_d = k, d
            k -= 1
        k = j
        while k < len(cand) and cand[k] - onset <= tol:
            d = cand[k] - onset
            if not flag[k] and d < best_d:
                best, best_d = k, d
            k += 1
        if best >= 0:
            flag[best] = True
            flags[i] = True
            matched += 1
    return flags, len(secondary) - matched


# ---------------------------------------------------------------------------
# Tuning / capo contradiction report
# ---------------------------------------------------------------------------


def capo_check(paths: VideoPaths) -> dict:
    """Surface tuning.json's verification block + recompute sub-floor counts.

    The recomputation reuses tuning.py's calibrated `_subfloor_evidence`
    (junk floor, sustain minimum, veto thresholds) against the CURRENT notes
    files, so a tuning.json written before the transcription existed — or
    before wave 1 added the verification block — still gets checked here.
    """
    tuning_data = _load_json(paths.tuning_json)
    if not isinstance(tuning_data, dict):
        return {"present": False}
    out: dict = {
        "present": True,
        "label": tuning_data.get("label"),
        "capo": tuning_data.get("capo"),
        "confidence": tuning_data.get("confidence"),
        "source": tuning_data.get("source"),
        # Wave-1 block; None on artifacts cached before it existed.
        "tuning_verification": tuning_data.get("verification"),
    }
    strings_midi = tuning_data.get("strings_midi")
    capo = tuning_data.get("capo")
    loaded = load_transcribed_notes(paths)
    if not isinstance(strings_midi, list) or not strings_midi or capo is None or loaded is None:
        out["recomputed"] = None
        return out
    notes, notes_source = loaded
    effective_low = int(min(strings_midi)) + int(capo)
    ev = _subfloor_evidence(notes, effective_low)
    out["recomputed"] = {
        "notes_source": notes_source,
        "notes_examined": len(notes),
        "effective_low_midi": effective_low,
        "subfloor_count": ev.all_count,
        "subfloor_sustained_count": ev.sustained_count,
        "subfloor_total_s": round(ev.total_s, 2),
        "subfloor_fraction": round(ev.fraction, 4),
        "would_veto": ev.veto,
    }
    return out


# ---------------------------------------------------------------------------
# Section scores
# ---------------------------------------------------------------------------


def band_for_score(score: float) -> str:
    if score >= SECTION_BAND_SOLID_MIN:
        return BAND_SOLID
    if score < SECTION_BAND_BAD_MAX:
        return BAND_BAD
    return BAND_SUSPECT


def _section_report(
    section: dict,
    notes: list[dict],
    assessments: list[tuple[float | None, str]],
    agreement: list[str | None],
    events: list[tuple[float, float, int] | None],
    stem_path: Path,
) -> dict | None:
    canonical = _pick_canonical_instance(section)
    if canonical is None:
        return None
    start, end = float(canonical["start"]), float(canonical["end"])
    idx = [i for i, n in enumerate(notes) if start <= float(n["start"]) < end]
    scored = [assessments[i][0] for i in idx if assessments[i][0] is not None]
    supported = sum(1 for i in idx if assessments[i][1] == VERDICT_SUPPORTED)
    agree_known = [agreement[i] for i in idx if agreement[i] is not None]
    section_events = [events[i] for i in idx if events[i] is not None]
    if section_events:
        score = salience.section_score(section_events, stem_path, start, end)
        band = band_for_score(score)
    else:
        score = None
        band = BAND_NO_NOTES
    return {
        "label": section.get("label"),
        "window": [start, end],
        "demo_quality": canonical.get("demo_quality"),
        "n_notes": len(idx),
        "salience_mean": round(sum(scored) / len(scored), 4) if scored else None,
        "pct_supported": round(supported / len(idx), 4) if idx else None,
        "agreement_rate": (
            round(sum(1 for a in agree_known if a == AGREEMENT_BOTH) / len(agree_known), 4)
            if agree_known
            else None
        ),
        "section_score": round(score, 4) if score is not None else None,
        "band": band,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def verification_input_paths(paths: VideoPaths) -> list[Path]:
    """Every artifact the verification report is computed from. Single source
    of truth shared with doctor's freshness check, so a verification.json
    older than ANY of these reads as stale (not just older than frets.json)."""
    return [
        paths.frets_json,
        paths.notes_mt3_json,
        paths.notes_json,
        paths.sections_json,
        paths.tuning_json,
        paths.guitar_stem,
    ]


def _generated_at(input_paths: list[Path]) -> str:
    """Timestamp derived from the newest INPUT mtime, not the wall clock, so
    re-running verify on unchanged inputs reproduces the identical report."""
    mtimes = [p.stat().st_mtime for p in input_paths if p.exists()]
    newest = max(mtimes) if mtimes else 0.0
    return datetime.fromtimestamp(newest, tz=UTC).isoformat()


def _input_echo(paths: VideoPaths) -> dict:
    echo: dict = {}
    for name, p in (
        ("frets_json", paths.frets_json),
        ("notes_mt3_json", paths.notes_mt3_json),
        ("notes_json", paths.notes_json),
        ("sections_json", paths.sections_json),
        ("tuning_json", paths.tuning_json),
        ("guitar_stem", paths.guitar_stem),
    ):
        if not p.exists():
            echo[name] = None
            continue
        entry: dict = {"path": str(p), "mtime": round(p.stat().st_mtime, 3)}
        echo[name] = entry
    # Provenance echoes so the report says WHAT it verified.
    mt3 = _load_json(paths.notes_mt3_json)
    if isinstance(mt3, dict) and echo.get("notes_mt3_json"):
        for key in ("backend", "filter", "provenance"):
            if key in mt3:
                echo["notes_mt3_json"][key] = mt3[key]
    tuning_data = _load_json(paths.tuning_json)
    if isinstance(tuning_data, dict) and echo.get("tuning_json"):
        for key in ("label", "capo", "source"):
            if key in tuning_data:
                echo["tuning_json"][key] = tuning_data[key]
    return echo


def verify(paths: VideoPaths, out_path: Path | None = None) -> dict:
    """Build + write the verification report. Returns the report dict.

    ``out_path`` overrides cache/<id>/verification.json — used by tests and
    read-only experiments against the cache.
    """
    frets = _load_json(paths.frets_json)
    if not isinstance(frets, dict) or not isinstance(frets.get("notes"), list):
        raise VerifyError(f"frets.json missing or unreadable at {paths.frets_json} — run frets")
    if not paths.guitar_stem.exists():
        raise VerifyError(f"guitar stem missing at {paths.guitar_stem} — run separate")

    notes: list[dict] = frets["notes"]
    tuning_midis = (frets.get("tuning") or {}).get("low_to_high_midi")
    pitches = [_note_pitch(n, tuning_midis) for n in notes]

    # 1. Per-note salience verdicts (chunked shared CQT).
    assessments = assess_all_notes(notes, pitches, paths.guitar_stem)

    # 2. Cross-model agreement. The tab's default source is MT3 (fret.py
    # backend default), so frets notes match against basic-pitch as the
    # second opinion; 'bp-only' counts basic-pitch notes no tab note claims.
    # Notes with underivable pitch are excluded from matching entirely (their
    # agreement stays null and 'pitch_unknown' counts them) instead of
    # entering with a sentinel pitch that would mislabel them 'mt3-only'.
    bp_events = _load_notes_list(paths.notes_json)
    agreement: list[str | None] = [None] * len(notes)
    if bp_events is None:
        agreement_summary: dict = {
            "checked": False,
            "reason": "notes.json (basic-pitch) not available",
        }
    elif not bp_events:
        agreement_summary = {
            "checked": False,
            "reason": "notes.json (basic-pitch) exists but contains no usable notes",
        }
    else:
        known = [i for i in range(len(notes)) if pitches[i] is not None]
        frets_events = [(float(notes[i]["start"]), pitches[i]) for i in known]
        flags, bp_unmatched = match_events(frets_events, bp_events)
        for f, i in zip(flags, known, strict=True):
            agreement[i] = AGREEMENT_BOTH if f else AGREEMENT_MT3_ONLY
        agreement_summary = {
            "checked": True,
            "onset_tolerance_s": AGREEMENT_ONSET_TOL_S,
            AGREEMENT_BOTH: sum(flags),
            AGREEMENT_MT3_ONLY: len(flags) - sum(flags),
            AGREEMENT_BP_ONLY: bp_unmatched,
            "pitch_unknown": len(notes) - len(known),
        }

    # 3. Tuning / capo contradiction report.
    capo = capo_check(paths)

    # 4. Section scores over render's canonical instances.
    events: list[tuple[float, float, int] | None] = [
        (float(n["start"]), float(n["end"]) - float(n["start"]), p) if p is not None else None
        for n, p in zip(notes, pitches, strict=True)
    ]
    sections_data = _load_json(paths.sections_json)
    per_section: list[dict] = []
    if isinstance(sections_data, dict) and isinstance(sections_data.get("sections"), list):
        for section in sections_data["sections"]:
            report = _section_report(
                section, notes, assessments, agreement, events, paths.guitar_stem
            )
            if report is not None:
                per_section.append(report)

    # 5. Summary + report.
    verdict_counts = {v: 0 for v in VERDICTS}
    for _, verdict in assessments:
        verdict_counts[verdict] += 1
    band_counts = {b: 0 for b in (BAND_SOLID, BAND_SUSPECT, BAND_BAD, BAND_NO_NOTES)}
    weighted = 0.0
    weight = 0
    for s in per_section:
        band_counts[s["band"]] += 1
        if s["section_score"] is not None:
            weighted += s["section_score"] * s["n_notes"]
            weight += s["n_notes"]
    overall_band = band_for_score(weighted / weight) if weight else "unknown"

    input_paths = verification_input_paths(paths)
    report = {
        "video_id": paths.video_id,
        "generated_at": _generated_at(input_paths),
        "inputs": _input_echo(paths),
        "per_note_columns": ["note_index", "salience", "verdict", "agreement"],
        "per_note": [
            [
                notes[i].get("note_index", i),
                round(assessments[i][0], 3) if assessments[i][0] is not None else None,
                assessments[i][1],
                agreement[i],
            ]
            for i in range(len(notes))
        ],
        "per_section": per_section,
        "summary": {
            "note_count": len(notes),
            "verdicts": verdict_counts,
            "agreement": agreement_summary,
            "section_bands": band_counts,
            "overall_band": overall_band,
            "capo_check": capo,
        },
    }

    target = out_path if out_path is not None else paths.verification_json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    return report
