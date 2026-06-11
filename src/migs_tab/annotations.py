"""Mine basic-pitch's detections for ornaments the primary backend (MT3) simplified.

We don't change the tab — instead we surface short hints that the user (or the
LLM-driven phase) can use to manually add ornaments like 12th-fret natural
harmonics, sustained lead notes, or ghost flourishes that MT3 dropped in
favor of a clean chord.

Two kinds of hints:
- **Harmonic candidate**: high pitch (≥72/C5) with a fundamental one octave
  lower visible in the primary notes around the same time. Classic acoustic
  natural-harmonic signature.
- **Sustained lead**: any pitch the secondary heard but the primary didn't,
  sustained ≥200ms, that isn't likely string-overtone noise.

Hints are deduped by pitch so 12 occurrences of the same harmonic across a
section yield one hint with a list of times.
"""

from __future__ import annotations

from collections import defaultdict

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Thresholds — empirically tuned on the Angie 30s slice. A pitch at or above
# C5 (MIDI 72) is rare on rhythm guitar accompaniment unless it's a harmonic
# or a deliberate lead; below 200ms duration is almost always overtone noise.
_HARMONIC_PITCH_THRESHOLD = 72
_LEAD_MIN_DURATION_S = 0.2
_ONSET_MATCH_TOL_S = 0.1
_OCTAVE_FUNDAMENTAL_WINDOW_S = 0.5
_MAX_HINTS_PER_SECTION = 5


def midi_to_name(midi: int) -> str:
    pc = midi % 12
    octave = midi // 12 - 1
    return f"{_PITCH_NAMES[pc]}{octave}"


def compute_section_hints(
    section_start: float,
    section_end: float,
    primary_notes: list[dict],
    secondary_notes: list[dict],
) -> list[str]:
    """Return prose hints about secondary-backend notes the primary missed.

    primary_notes / secondary_notes are the full unwindowed lists; we filter
    each to ``[section_start, section_end)`` internally.
    """
    p_in = [n for n in primary_notes if section_start <= n["start"] < section_end]
    s_in = [n for n in secondary_notes if section_start <= n["start"] < section_end]
    if not s_in:
        return []

    novel = [n for n in s_in if not _matches_primary(n, p_in)]
    if not novel:
        return []

    by_pitch: dict[int, list[dict]] = defaultdict(list)
    for n in novel:
        by_pitch[n["pitch"]].append(n)

    candidates: list[tuple[int, str]] = []
    for pitch, hits in by_pitch.items():
        hint = _classify_pitch_group(pitch, hits, p_in, section_start)
        if hint:
            # Sort key: harmonic candidates first (highest priority), then by pitch desc.
            priority = 0 if "harmonic" in hint else 1
            candidates.append(((priority, -pitch), hint))

    candidates.sort(key=lambda t: t[0])
    return [hint for _, hint in candidates[:_MAX_HINTS_PER_SECTION]]


def _matches_primary(secondary: dict, primary: list[dict]) -> bool:
    """Same pitch, onset within ±100ms."""
    s_start = secondary["start"]
    s_pitch = secondary["pitch"]
    for p in primary:
        if p["pitch"] == s_pitch and abs(p["start"] - s_start) <= _ONSET_MATCH_TOL_S:
            return True
    return False


def _classify_pitch_group(
    pitch: int,
    hits: list[dict],
    primary: list[dict],
    section_start: float,
) -> str | None:
    """Decide whether a group of same-pitch novel notes is worth a hint."""
    if pitch >= _HARMONIC_PITCH_THRESHOLD:
        fundamental = pitch - 12
        # Any primary note one octave lower within ±0.5s of ANY of the hits?
        has_fundamental = any(
            p["pitch"] == fundamental
            and abs(p["start"] - h["start"]) <= _OCTAVE_FUNDAMENTAL_WINDOW_S
            for h in hits
            for p in primary
        )
        times = sorted({round(h["start"] - section_start, 1) for h in hits})
        time_str = _format_times(times)
        if has_fundamental:
            return (
                f"natural harmonic candidate — {midi_to_name(pitch)} (basic-pitch heard "
                f"it at {time_str}; MT3 has the {midi_to_name(fundamental)} fundamental "
                f"nearby — likely a 12th-fret-harmonic ornament)"
            )
        # High lead note, no obvious fundamental — only flag if at least 2 hits
        # so we don't surface single-pitch noise spikes.
        if len(hits) >= 2:
            return (
                f"high lead — {midi_to_name(pitch)} at {time_str} (basic-pitch hears "
                f"this {len(hits)}× but MT3 dropped it; could be a melodic flourish)"
            )
        return None

    # Below C5: only flag sustained durations — but ONLY if the primary backend
    # has no note at this exact pitch anywhere in the section. If MT3 hit the
    # same pitch elsewhere in the section, the player is holding it as a chord
    # tone and basic-pitch's longer-duration reading isn't a "missed ornament"
    # worth surfacing.
    if any(p["pitch"] == pitch for p in primary):
        return None
    sustained = [h for h in hits if (h["end"] - h["start"]) >= _LEAD_MIN_DURATION_S]
    if len(sustained) >= 2:
        times = sorted({round(h["start"] - section_start, 1) for h in sustained})
        avg_dur = sum(h["end"] - h["start"] for h in sustained) / len(sustained)
        return (
            f"sustained — {midi_to_name(pitch)} at {_format_times(times)} "
            f"(avg {avg_dur:.2f}s; basic-pitch heard a held note MT3 simplified out)"
        )
    return None


def _format_times(times: list[float]) -> str:
    if len(times) == 1:
        return f"+{times[0]:.1f}s"
    if len(times) <= 4:
        return ", ".join(f"+{t:.1f}s" for t in times)
    return f"+{times[0]:.1f}s, +{times[1]:.1f}s, … (+{len(times) - 2} more)"
