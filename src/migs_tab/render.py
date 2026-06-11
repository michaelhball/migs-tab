"""Phase 4: render a section-by-section ASCII guitar tab.

Reads ``sections.json`` (which segments of the video correspond to which
parts of the song, grouped by repetition), picks the *canonical* instance
of each section (prefer slow-walkthroughs, then normal-tempo, then the
longest take), filters ``frets.json`` to that instance's time window,
applies any ``frets.overrides.json`` corrections from the vision pass,
and emits a 6-line ASCII tab grouped by section label.

For each section we also run librosa beat tracking on the guitar stem
slice corresponding to that section's canonical instance. Onset clusters
are snapped to the nearest 8th-note subdivision so the tab columns align
to beats — much easier to read than the event-ordered layout we had
before, and gives an implicit feel for the song's rhythm.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .annotations import compute_section_hints
from .paths import DEFAULT_OUTPUT_DIR, VideoPaths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default string letters for standard tuning (top = high E, bottom = low E).
# Derived dynamically from the detected tuning when non-standard.
_DEFAULT_TAB_STRING_LETTERS = ["e", "B", "G", "D", "A", "E"]
_PITCH_NAMES_NATURAL = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _tab_string_letters_for_tuning(strings_midi: list[int]) -> list[str]:
    """Top-to-bottom string labels (high E first) based on the open-string
    pitches of the tuning. Top-string letter is lowercased; bottom is upper.
    """
    if not strings_midi or len(strings_midi) != 6:
        return _DEFAULT_TAB_STRING_LETTERS
    # strings_midi is low → high; tab is high → low.
    labels: list[str] = []
    for i in range(5, -1, -1):
        name = _PITCH_NAMES_NATURAL[strings_midi[i] % 12]
        # Conventionally only the top (highest) string is lowercased to
        # disambiguate from the bottom string when both are E.
        if i == 5:
            name = name.lower()
        labels.append(name)
    return labels


# Wrap each tab system at this many characters.
_DEFAULT_LINE_WIDTH = 72

# If the gap between two consecutive onset clusters exceeds this many
# seconds, insert an extra "rest" column to suggest a pause to the reader.
_PAUSE_GAP_SECONDS = 0.8

# Sections whose chord_progression is empty are talking-only and skipped.
# Plus a hard exclusion list for sections we never want to render.
_SKIP_LABELS = {"closing_remarks"}

# Noise-reduction thresholds applied before rendering. basic-pitch is honest
# about every pitch it hears, including sympathetic resonance and short
# transients — these would clutter the tab without representing intentional
# notes the player meant to land.
_MIN_NOTE_DURATION = 0.08  # drop notes shorter than this (~ a single hop)
_MIN_NOTE_VELOCITY = 35  # drop notes quieter than this on the 0..127 scale
# Duplicate-onset suppression: same-pitch onsets within this window are
# re-detections of one pluck; the louder (then longer) of the pair is kept.
# Onset distance is the ONLY criterion — the old sustain filter (drop any
# onset inside the previous note's span) deleted genuine re-articulations,
# because MT3 emits generously long ringing durations: a 5s strummed-E
# demo rendered as ONE note.
_DEDUPE_WINDOW = 0.10

# Cross-instance voting: a canonical-instance note is kept only if its
# (chord_name, pitch_class) pair appears in at least this many *other*
# instances of the same section. 1 = "at least one other take confirms".
# Skipped entirely for sections with only one instance.
_CROSS_INSTANCE_MIN_SUPPORT = 1

# Cross-instance voting loud-note bypass: notes at/above this velocity are
# kept even with no support from another take.
_LOUD_VELOCITY_KEEP = 75
# frets.json notes carry NO velocity field today: MT3 emits constant
# velocity 100 at the raw MIDI level and the frets pipeline doesn't
# propagate basic-pitch's. Until the salience integration lands and
# populates an energy-proxy velocity, a missing velocity must mean "keep"
# — hence a sentinel above any real 0..127 value.
_VELOCITY_WHEN_ABSENT = 999

# Demo-quality ranking when picking a canonical instance.
_DEMO_QUALITY_RANK = {
    "slow-walkthrough": 3,
    "normal-tempo": 2,
    "repeated-loop": 2,
    "partial": 1,
}

# Cap per-section layout footnotes; beyond this they collapse into a
# single "+N more" line so collision-heavy sections don't bury the tab
# (mirrors annotations._MAX_HINTS_PER_SECTION).
_MAX_LAYOUT_NOTES_PER_SECTION = 8

# When a cluster snaps onto an occupied subdivision slot, scan forward at
# most this many slots for a free one (3 slots = 1.5 beats at 8th-note
# granularity). Drifting an onset further than that misrepresents the
# rhythm worse than a shared cell does — beyond the budget the clusters
# share the cell and a footnote is emitted.
_MAX_SLOT_SHIFT = 3

# Beat-tracking + quantization parameters.
_BEAT_SR = 22050
_SUBDIVISIONS_PER_BEAT = 2  # 8th notes; bump to 4 for 16th-note quantization
_BEATS_PER_BAR = 4  # most acoustic guitar tutorials are in 4/4
# Fallback tempo to use when beat tracking fails / yields no beats.
_FALLBACK_TEMPO_BPM = 90.0
# Plausible tempo range for acoustic guitar. librosa.beat_track sometimes
# locks onto twice or half the real tempo; we halve or double until the
# result lands in this range.
_TEMPO_PLAUSIBLE_MIN = 55.0
_TEMPO_PLAUSIBLE_MAX = 165.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class RenderedSection:
    label: str
    description: str
    canonical_start: float
    canonical_end: float
    chord_progression: list[str]
    cluster_count: int
    note_count: int
    ascii_tab: str
    tempo_bpm: float
    hints: list[str] = field(default_factory=list)
    # Beat-grid slot collisions: how many clusters snapped onto an already-
    # occupied subdivision slot, plus footnotes for same-string drops and
    # cross-string merges the forward scan could not resolve.
    slot_collisions: int = 0
    layout_notes: list[str] = field(default_factory=list)


@dataclass
class TuningInfo:
    label: str
    capo: int
    strings_midi: list[int]
    source: str
    confidence: float
    string_letters: list[str]  # top-down (high E first)

    @classmethod
    def from_paths(cls, paths: VideoPaths) -> TuningInfo:
        if paths.tuning_json.exists():
            try:
                data = json.loads(paths.tuning_json.read_text())
                return cls(
                    label=data.get("label", "Standard"),
                    capo=int(data.get("capo", 0)),
                    strings_midi=list(data.get("strings_midi", [40, 45, 50, 55, 59, 64])),
                    source=data.get("source", "default"),
                    confidence=float(data.get("confidence", 0.0)),
                    string_letters=_tab_string_letters_for_tuning(
                        data.get("strings_midi", [40, 45, 50, 55, 59, 64])
                    ),
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return cls(
            label="Standard",
            capo=0,
            strings_midi=[40, 45, 50, 55, 59, 64],
            source="default",
            confidence=0.0,
            string_letters=list(_DEFAULT_TAB_STRING_LETTERS),
        )


def render(
    paths: VideoPaths,
    output_root: Path = DEFAULT_OUTPUT_DIR,
    line_width: int = _DEFAULT_LINE_WIDTH,
    force: bool = False,
) -> Path:
    """Render the section-by-section tab. Returns the tab.txt path.

    Prefers ``sections.json`` (rich LLM-labeled sections) when present.
    Falls back to deriving a flat list of "sections" from ``structure.json``
    when sections.json is missing — useful for looser videos (live
    recordings, performances without instructor commentary) where the
    Claude-skill Phase 2 step hasn't run.
    """
    if not paths.frets_json.exists():
        raise FileNotFoundError(
            f"frets.json not found at {paths.frets_json}; run `migs-tab frets` first."
        )

    out_dir = paths.output_dir(output_root)
    tab_path = out_dir / "tab.txt"
    md_path = out_dir / "tab.md"
    # Reuse existing outputs only when they are at least as new as every
    # input artifact — a bare existence check let users keep reading stale
    # tabs after upstream fixes re-wrote frets.json / tuning.json / etc.
    if not force and _outputs_fresh(paths, tab_path, md_path):
        return tab_path

    if paths.sections_json.exists():
        sections_data = json.loads(paths.sections_json.read_text())
    elif paths.structure_json.exists():
        sections_data = _sections_from_structure(paths.structure_json)
    else:
        raise FileNotFoundError(
            "Neither sections.json nor structure.json is available; run `migs-tab structure` first."
        )
    frets_data = json.loads(paths.frets_json.read_text())
    overrides = _load_overrides(paths)
    notes = _apply_overrides(frets_data["notes"], overrides)
    notes = _filter_noise(notes)

    tuning_info = TuningInfo.from_paths(paths)

    # Vision-verified chord shapes — for each note whose chord context has a
    # verified (string, fret) voicing, override the algorithm's assignment.
    chord_spans = _load_chord_spans_for_render(paths)
    verified_shapes = _load_verified_chord_shapes(paths)
    notes = _apply_verified_chord_shapes(notes, chord_spans, verified_shapes, tuning_info)

    # Secondary-backend notes for ornament hints. If MT3 drove the tab, the
    # secondary is basic-pitch and vice versa. Missing = no hints, fine.
    secondary_notes = _load_secondary_notes_for_hints(paths)

    # Build cross-instance pitch-class support per section, so we can drop
    # canonical-instance notes that no other take confirms.
    cross_support = _build_cross_instance_support(sections_data, notes, chord_spans)

    # Track per-section data needed by the MusicXML exporter too, so the
    # two output formats stay in lockstep (both consume the same beat grid
    # + filtered notes).
    notes_by_section: dict[str, list[dict]] = {}
    beat_times_by_section: dict[str, list[float]] = {}

    rendered: list[RenderedSection] = []
    for section in sections_data["sections"]:
        if section["label"] in _SKIP_LABELS:
            continue
        if not section.get("chord_progression"):
            continue
        canonical = _pick_canonical_instance(section)
        if canonical is None:
            continue
        section_notes = [n for n in notes if canonical["start"] <= n["start"] < canonical["end"]]
        if not section_notes:
            continue
        section_notes = _apply_cross_instance_support(
            section_notes,
            chord_spans,
            cross_support.get(section["label"], {}),
            cross_support_min=_CROSS_INSTANCE_MIN_SUPPORT,
            instance_count=len(section.get("instances", [])),
        )
        if not section_notes:
            continue
        tempo_bpm, beat_times = _detect_beats(
            paths.guitar_stem, canonical["start"], canonical["end"]
        )
        notes_by_section[section["label"]] = section_notes
        beat_times_by_section[section["label"]] = beat_times
        ascii_tab, slot_collisions, layout_notes = _render_section_tab(
            section_notes,
            line_width=line_width,
            beat_times=beat_times,
            string_letters=tuning_info.string_letters,
        )
        clusters_in_section = {n["cluster_id"] for n in section_notes}
        hints = (
            compute_section_hints(
                canonical["start"], canonical["end"], section_notes, secondary_notes
            )
            if secondary_notes
            else []
        )
        rendered.append(
            RenderedSection(
                label=section["label"],
                description=section.get("description", ""),
                canonical_start=canonical["start"],
                canonical_end=canonical["end"],
                chord_progression=section.get("chord_progression", []),
                cluster_count=len(clusters_in_section),
                note_count=len(section_notes),
                ascii_tab=ascii_tab,
                tempo_bpm=tempo_bpm,
                hints=hints,
                slot_collisions=slot_collisions,
                layout_notes=layout_notes,
            )
        )

    tab_text = _format_full_tab(sections_data, rendered, tuning_info)
    tab_path.write_text(tab_text)
    md_path.write_text(_format_markdown(sections_data, rendered, tuning_info))

    # Also emit the chord-chart-only view (always — cheap to compute and
    # useful for loose / live videos where the note-level tab is noisy).
    if paths.structure_json.exists():
        chart_text = _render_chord_chart(paths, tuning_info)
        (out_dir / "chord-chart.txt").write_text(chart_text)

    # MusicXML export — opens in MuseScore / TuxGuitar / Guitar Pro.
    from . import musicxml as musicxml_mod

    xml_bytes = musicxml_mod.render_musicxml(
        sections_data=sections_data,
        rendered=rendered,
        tuning=tuning_info,
        notes_by_section=notes_by_section,
        beat_times_by_section=beat_times_by_section,
        subdivisions_per_beat=_SUBDIVISIONS_PER_BEAT,
        beats_per_bar=_BEATS_PER_BAR,
    )
    (out_dir / "tab.musicxml").write_bytes(xml_bytes)
    return tab_path


# ---------------------------------------------------------------------------
# Output freshness
# ---------------------------------------------------------------------------


def _render_input_paths(paths: VideoPaths) -> list[Path]:
    """Every cache artifact render() consumes. Optional ones (overrides,
    verified shapes, tuning, secondary notes) are listed too — when they
    appear or change, the rendered tab must change with them."""
    return [
        paths.frets_json,
        paths.sections_json,
        paths.structure_json,
        paths.frets_overrides_json,
        paths.chord_shapes_verified_json,
        paths.tuning_json,
        paths.notes_json,  # secondary backend, drives the ornament hints
        paths.guitar_stem,  # beat grids via _detect_beats
    ]


def _newest_input_mtime(paths: VideoPaths) -> float:
    mtimes = [p.stat().st_mtime for p in _render_input_paths(paths) if p.exists()]
    return max(mtimes) if mtimes else 0.0


def _outputs_fresh(paths: VideoPaths, tab_path: Path, md_path: Path) -> bool:
    """True when both rendered outputs exist and are at least as new as
    every input artifact, i.e. it is safe to skip re-rendering."""
    if not (tab_path.exists() and md_path.exists()):
        return False
    output_mtime = min(tab_path.stat().st_mtime, md_path.stat().st_mtime)
    return output_mtime >= _newest_input_mtime(paths)


# ---------------------------------------------------------------------------
# Vision-verified chord-shape overrides
# ---------------------------------------------------------------------------


def _load_verified_chord_shapes(paths: VideoPaths) -> dict:
    """Load chord-shapes-verified.json if present. The structured format is:

    {
      "video_id": "...",
      "verified": {
        "<chord_name>": {
          "voicing": [
            {"string": <0..5>, "fret": <int>, "midi_pitch": <int, optional>},
            ...
          ],
          "applies_to": "all_spans"  |  [{"start": ..., "end": ...}, ...],
          "notes": "..."   # optional
        }
      }
    }

    ``midi_pitch`` is optional-but-recommended: when absent it is derived
    from the detected tuning as ``strings_midi[string] + capo + fret``
    (frets.json frets are capo-relative, pitches are sounding pitches).

    Older narrative-style files (without an explicit ``voicing`` list per
    chord) are loaded but have no effect — we ignore entries that don't
    carry a structured voicing.
    """
    if not paths.chord_shapes_verified_json.exists():
        return {}
    try:
        return json.loads(paths.chord_shapes_verified_json.read_text())
    except json.JSONDecodeError:
        return {}


def _apply_verified_chord_shapes(
    notes: list[dict],
    chord_spans: list[tuple[float, float, str]],
    verified_data: dict,
    tuning: TuningInfo | None = None,
) -> list[dict]:
    """Override (string, fret) for notes whose chord context has a verified
    voicing for their pitch.

    For each note: look up the chord active at the note's start time. If
    the verified data has a structured ``voicing`` for that chord AND the
    voicing covers this exact MIDI pitch AND the note's time falls inside
    the verified entry's ``applies_to`` scope, replace the algorithm's
    (string, fret) with the verified ones.

    ``midi_pitch`` is optional in voicing entries: when absent it is
    derived from the tuning (``strings_midi[string] + capo + fret``), so
    the SKILL.md-documented {string, fret} format works. Malformed entries
    emit a UserWarning instead of being silently dropped — a verified file
    that silently does nothing is worse than a noisy one.
    """
    if not verified_data or not chord_spans:
        return notes
    verified_map = verified_data.get("verified", {})
    if not verified_map:
        return notes

    # Pre-extract structured-voicing pitch lookups per chord.
    by_chord: dict[str, dict[int, tuple[int, int]]] = {}
    applies_to_by_chord: dict[str, list[dict] | str] = {}
    for chord, entry in verified_map.items():
        voicing = entry.get("voicing")
        if voicing is None:
            continue  # narrative-style entry, intentionally inert
        if not isinstance(voicing, list):
            warnings.warn(
                f"chord-shapes-verified: '{chord}' has a non-list 'voicing' "
                f"({type(voicing).__name__}); entry ignored.",
                stacklevel=2,
            )
            continue
        lookup: dict[int, tuple[int, int]] = {}
        for v in voicing:
            try:
                string = int(v["string"])
                fret = int(v["fret"])
            except (KeyError, TypeError, ValueError):
                warnings.warn(
                    f"chord-shapes-verified: '{chord}' voicing entry {v!r} is "
                    "missing a valid 'string'/'fret'; entry ignored.",
                    stacklevel=2,
                )
                continue
            derived = _derive_midi_pitch(tuning, string, fret)
            declared = v.get("midi_pitch")
            if declared is not None:
                try:
                    pitch = int(declared)
                except (TypeError, ValueError):
                    warnings.warn(
                        f"chord-shapes-verified: '{chord}' voicing entry {v!r} "
                        f"has a non-integer 'midi_pitch'; entry ignored.",
                        stacklevel=2,
                    )
                    continue
                if derived is not None and pitch != derived:
                    warnings.warn(
                        f"chord-shapes-verified: '{chord}' voicing entry {v!r} "
                        f"declares midi_pitch {pitch} but string {string} fret "
                        f"{fret} sounds as {derived} under the detected tuning "
                        "— the declared pitch is used, but check the entry.",
                        stacklevel=2,
                    )
            elif derived is not None:
                pitch = derived
            else:
                warnings.warn(
                    f"chord-shapes-verified: '{chord}' voicing entry {v!r} has "
                    "no 'midi_pitch' and no tuning is available to derive it; "
                    "entry ignored.",
                    stacklevel=2,
                )
                continue
            lookup[pitch] = (string, fret)
        if lookup:
            by_chord[chord] = lookup
            applies_to_by_chord[chord] = entry.get("applies_to", "all_spans")

    if not by_chord:
        return notes

    out: list[dict] = []
    for n in notes:
        chord = _chord_for_time(chord_spans, n["start"])
        applied = False
        if chord and chord in by_chord:
            applies = applies_to_by_chord[chord]
            if applies == "all_spans" or _in_any_time_range(applies, n["start"]):
                target = by_chord[chord].get(int(n["pitch"]))
                if target is not None:
                    replacement = dict(n)
                    replacement["string"] = target[0]
                    replacement["fret"] = target[1]
                    replacement["overridden_by"] = f"verified-shape:{chord}"
                    out.append(replacement)
                    applied = True
        if not applied:
            out.append(n)
    return out


def _derive_midi_pitch(tuning: TuningInfo | None, string: int, fret: int) -> int | None:
    """Sounding MIDI pitch of (string, fret) under the detected tuning.

    frets.json frets are relative to the capo and pitches are sounding
    pitches, so: ``strings_midi[string] + capo + fret`` (matches
    fret._load_tuning, which adds the capo to every open string).
    Returns None when the tuning is unavailable or the string index is
    out of range.
    """
    if tuning is None or not (0 <= string < len(tuning.strings_midi)):
        return None
    return tuning.strings_midi[string] + tuning.capo + fret


def _in_any_time_range(ranges, t: float) -> bool:
    for r in ranges or []:
        try:
            if float(r["start"]) <= t < float(r["end"]):
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# Chord-chart-only render mode
# ---------------------------------------------------------------------------

# Skip chord spans shorter than this when rendering the chord chart —
# they're usually basic-pitch artifacts or quick passing chords that
# aren't useful in a high-level chord chart.
_CHORD_CHART_MIN_DURATION = 0.5


def _format_time(seconds: float) -> str:
    """Render seconds as M:SS for chord-chart timestamps."""
    if seconds < 0:
        seconds = 0
    minutes = int(seconds // 60)
    secs = int(seconds - minutes * 60)
    return f"{minutes}:{secs:02d}"


def _render_chord_chart(paths: VideoPaths, tuning: TuningInfo) -> str:
    """Render a simple time → chord listing for the whole video.

    Reads structure.json directly (independent of frets.json), so it
    works for any video the pipeline has at least run structure on.
    Useful for live performances where the 6-line tab is too noisy to
    be worth reading but the chord progression is still solid.
    """
    if not paths.structure_json.exists():
        return ""
    data = json.loads(paths.structure_json.read_text())

    lines: list[str] = []
    video_id = data.get("video_id", "?")
    lines.append(f"# Chord chart — video {video_id}")
    lines.append("")
    lines.append(f"Tuning: {tuning.label}")
    if tuning.capo:
        lines.append(
            f"Capo: fret {tuning.capo}  "
            "(chord names below are the SOUNDING chords; to play with "
            f"your capo at fret {tuning.capo}, transpose each name down "
            f"{tuning.capo} semitones to get the shape to finger)"
        )
    lines.append("")

    segments = data.get("playing_segments", [])
    if not segments:
        lines.append("_No playing segments detected._")
        return "\n".join(lines) + "\n"

    lines.append(f"Detected {len(segments)} playing segment(s):")
    lines.append("")

    for seg in segments:
        seg_start = _format_time(seg["start"])
        seg_end = _format_time(seg["end"])
        lines.append("=" * 60)
        lines.append(f"Segment {seg['id']}   {seg_start} – {seg_end}  ({seg['duration']:.1f}s)")
        chords = [
            c for c in seg.get("chords", []) if (c["end"] - c["start"]) >= _CHORD_CHART_MIN_DURATION
        ]
        if not chords:
            lines.append("  (no chord spans long enough to chart)")
            lines.append("")
            continue
        # Collapse consecutive same-chord spans (sometimes chroma flickers).
        collapsed: list[dict] = []
        for c in chords:
            if collapsed and collapsed[-1]["chord"] == c["chord"]:
                collapsed[-1]["end"] = c["end"]
            else:
                collapsed.append(dict(c))

        lines.append("")
        lines.append(f"  {'time':<7} {'chord':<8} duration")
        lines.append(f"  {'-' * 7} {'-' * 8} --------")
        for c in collapsed:
            dur = c["end"] - c["start"]
            lines.append(f"  {_format_time(c['start']):<7} {c['chord']:<8} {dur:5.1f}s")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# sections.json fallback (for looser videos)
# ---------------------------------------------------------------------------


def _sections_from_structure(structure_json_path: Path) -> dict:
    """Derive a flat sections.json-shaped object from structure.json.

    Each playing segment becomes a single section labeled `segment_<id>`
    with one instance (the segment itself) at "demo_quality: normal-tempo".
    Chord progression is the segment's chord list. This means render() can
    work on videos that never had an LLM-driven Phase 2 labeling pass.
    """
    data = json.loads(structure_json_path.read_text())
    sections: list[dict] = []
    for seg in data.get("playing_segments", []):
        chords = [c["chord"] for c in seg.get("chords", []) if (c["end"] - c["start"]) > 0.3]
        if not chords:
            continue
        sections.append(
            {
                "label": f"segment_{seg['id']:02d}",
                "description": (
                    f"Playing segment {seg['id']}: "
                    f"{seg['start']:.1f}-{seg['end']:.1f}s "
                    f"({seg['duration']:.1f}s). "
                    "Auto-derived from audio analysis (no LLM section labeling)."
                ),
                "chord_progression": chords,
                "instances": [
                    {
                        "segment_id": seg["id"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "demo_quality": "normal-tempo",
                    }
                ],
            }
        )
    return {
        "video_id": data.get("video_id", "?"),
        "structural_summary": (
            f"Auto-derived from structure.json — {len(sections)} playing "
            f"segment(s), no LLM labeling pass."
        ),
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Cross-instance voting
# ---------------------------------------------------------------------------


def _load_secondary_notes_for_hints(paths: VideoPaths) -> list[dict]:
    """Read notes from whichever backend did NOT drive the tab, for ornament hints.

    Currently we always use basic-pitch's notes.json as the secondary. The
    frets pipeline reads MT3 by default; basic-pitch's note list lives at
    paths.notes_json. Missing → return [] (no hints emitted).
    """
    if not paths.notes_json.exists():
        return []
    try:
        data = json.loads(paths.notes_json.read_text())
        notes = data.get("notes", [])
        # Defensive: tolerate either basic-pitch or MT3 schema.
        return [n for n in notes if "start" in n and "pitch" in n and "end" in n]
    except (OSError, json.JSONDecodeError):
        return []


def _load_chord_spans_for_render(paths: VideoPaths) -> list[tuple[float, float, str]]:
    """Same shape as fret._load_chord_spans, loaded here for cross-instance
    voting. Returns [] if structure.json is missing."""
    if not paths.structure_json.exists():
        return []
    data = json.loads(paths.structure_json.read_text())
    spans: list[tuple[float, float, str]] = []
    for seg in data.get("playing_segments", []):
        for c in seg.get("chords", []):
            spans.append((float(c["start"]), float(c["end"]), c["chord"]))
    spans.sort(key=lambda x: x[0])
    return spans


def _chord_for_time(spans: list[tuple[float, float, str]], t: float) -> str | None:
    for s, e, name in spans:
        if s <= t < e:
            return name
        if s > t:
            return None
    return None


def _build_cross_instance_support(
    sections_data: dict,
    notes: list[dict],
    chord_spans: list[tuple[float, float, str]],
) -> dict[str, dict[tuple[str, int], int]]:
    """For each section, count *how many of its instances* contain each
    (chord_name, MIDI pitch) pair (exact octave).

    Returns ``{section_label: {(chord, midi_pitch): count}}``. We vote
    on exact pitch (not pitch class) so an octave-misidentification
    by basic-pitch on the canonical take gets caught when no other
    instance reports the same pitch in the same chord context.
    """
    by_label: dict[str, dict[tuple[str, int], int]] = {}
    notes_sorted = sorted(notes, key=lambda n: n["start"])

    for section in sections_data.get("sections", []):
        instances = section.get("instances", [])
        if len(instances) < 2:
            continue
        per_pair_supports: dict[tuple[str, int], int] = {}
        for inst in instances:
            seen_in_instance: set[tuple[str, int]] = set()
            for n in notes_sorted:
                if n["start"] >= inst["end"]:
                    break
                if n["start"] < inst["start"]:
                    continue
                chord = _chord_for_time(chord_spans, n["start"])
                if chord is None:
                    continue
                seen_in_instance.add((chord, n["pitch"]))
            for pair in seen_in_instance:
                per_pair_supports[pair] = per_pair_supports.get(pair, 0) + 1
        by_label[section["label"]] = per_pair_supports
    return by_label


def _apply_cross_instance_support(
    section_notes: list[dict],
    chord_spans: list[tuple[float, float, str]],
    pair_supports: dict[tuple[str, int], int],
    cross_support_min: int,
    instance_count: int,
) -> list[dict]:
    """Drop notes whose exact (chord, pitch) is in fewer than
    ``cross_support_min + 1`` instances. Loud notes (velocity ≥
    ``_LOUD_VELOCITY_KEEP``) survive even without cross-instance backup —
    the canonical's confidence on its own is decisive."""
    if instance_count < 2 or not pair_supports:
        return section_notes
    threshold = cross_support_min + 1  # canonical counts as one
    survivors: list[dict] = []
    for n in section_notes:
        chord = _chord_for_time(chord_spans, n["start"])
        if chord is None:
            survivors.append(n)
            continue
        pair = (chord, n["pitch"])
        support = pair_supports.get(pair, 0)
        if support >= threshold:
            survivors.append(n)
        # Velocity is absent from frets.json today (see _VELOCITY_WHEN_ABSENT)
        # so this gate keeps every unsupported note until the salience
        # integration populates energy-proxy velocities. Do not remove it.
        elif n.get("velocity", _VELOCITY_WHEN_ABSENT) >= _LOUD_VELOCITY_KEEP:
            survivors.append(n)
    return survivors


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------


def _refine_tempo_octave(tempo: float, beats: list[float]) -> tuple[float, list[float]]:
    """Halve or double the tempo (and adjust beats accordingly) until it
    lands in a plausible acoustic-guitar range.

    librosa.beat.beat_track occasionally locks onto 2× or ½× the actual
    tempo — detecting eighth notes as quarters, or half notes as quarters.
    For each octave-wrong case there's a deterministic fix:
     - tempo too FAST (>165): halve, taking every other detected beat.
     - tempo too SLOW (<55): double, inserting midpoints between beats.

    Loops until the tempo is in range or we run out of headroom.
    """
    # Halve while too fast.
    while tempo > _TEMPO_PLAUSIBLE_MAX and len(beats) >= 4:
        tempo = tempo / 2
        beats = beats[::2]
    # Double while too slow.
    while tempo < _TEMPO_PLAUSIBLE_MIN and len(beats) >= 2:
        doubled: list[float] = []
        for i in range(len(beats) - 1):
            doubled.append(beats[i])
            doubled.append((beats[i] + beats[i + 1]) / 2.0)
        doubled.append(beats[-1])
        tempo = tempo * 2
        beats = doubled
    return tempo, beats


def _detect_beats(audio_path: Path, start_s: float, end_s: float) -> tuple[float, list[float]]:
    """Run librosa beat tracking on the [start, end] slice of the guitar stem.

    Returns (tempo_bpm, absolute beat times in the video's time frame).
    Falls back to a uniform grid at ``_FALLBACK_TEMPO_BPM`` if librosa
    finds no beats (very rare on guitar-stem audio). The detected tempo
    is also refined toward the plausible 55-165 bpm range — librosa
    sometimes returns 2× or ½× the real tempo.
    """
    if not audio_path.exists() or end_s <= start_s:
        return _FALLBACK_TEMPO_BPM, _uniform_beat_grid(start_s, end_s, _FALLBACK_TEMPO_BPM)

    duration = end_s - start_s
    try:
        y, sr = librosa.load(
            str(audio_path), sr=_BEAT_SR, offset=start_s, duration=duration, mono=True
        )
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        beat_times_rel = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = [start_s + float(t) for t in beat_times_rel]
    except Exception:
        return _FALLBACK_TEMPO_BPM, _uniform_beat_grid(start_s, end_s, _FALLBACK_TEMPO_BPM)

    if not beat_times:
        return _FALLBACK_TEMPO_BPM, _uniform_beat_grid(start_s, end_s, _FALLBACK_TEMPO_BPM)

    # Extend the grid to cover [start_s, end_s] using the detected period —
    # librosa often misses the first beat or two and the trailing ones.
    if len(beat_times) >= 2:
        period = beat_times[1] - beat_times[0]
    else:
        period = 60.0 / tempo_val if tempo_val > 0 else 60.0 / _FALLBACK_TEMPO_BPM

    extended: list[float] = list(beat_times)
    t = beat_times[0] - period
    while t > start_s - period / 2:
        extended.insert(0, t)
        t -= period
    t = beat_times[-1] + period
    while t < end_s + period / 2:
        extended.append(t)
        t += period
    # Octave-correct the tempo if librosa returned a 2× or ½× answer.
    return _refine_tempo_octave(tempo_val, extended)


def _uniform_beat_grid(start_s: float, end_s: float, bpm: float) -> list[float]:
    period = 60.0 / bpm
    return [start_s + i * period for i in range(int((end_s - start_s) / period) + 1)]


def _subdivisions_from_beats(beat_times: list[float]) -> list[tuple[float, bool]]:
    """Expand beat times into a list of (time, is_beat) at the chosen
    subdivisions-per-beat granularity. is_beat=True marks downbeats."""
    if len(beat_times) < 2:
        return [(t, True) for t in beat_times]
    grid: list[tuple[float, bool]] = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        step = (beat_end - beat_start) / _SUBDIVISIONS_PER_BEAT
        for j in range(_SUBDIVISIONS_PER_BEAT):
            grid.append((beat_start + j * step, j == 0))
    grid.append((beat_times[-1], True))
    return grid


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------


def _filter_noise(notes: list[dict]) -> list[dict]:
    """Drop short, quiet, or duplicate-onset notes."""
    survivors: list[dict] = []
    for n in notes:
        dur = n.get("end", n["start"]) - n["start"]
        if dur < _MIN_NOTE_DURATION:
            continue
        # Quiet-note floor. frets.json notes carry NO velocity field today
        # (MT3 emits constant velocity 100), so this gate is a no-op until
        # the salience integration lands and adds energy-proxy velocities.
        # Keep it: basic-pitch-backed runs DO have real velocities.
        vel = n.get("velocity")
        if vel is not None and vel < _MIN_NOTE_VELOCITY:
            continue
        survivors.append(n)

    # Duplicate-onset suppression: same-pitch onsets within _DEDUPE_WINDOW
    # of the last kept same-pitch onset are re-detections of one pluck —
    # keep the louder (then longer) of the pair, NOT necessarily the first.
    # Same-pitch onsets further apart than the window always survive, so
    # re-picking a ringing string or re-strumming a chord is never dropped.
    survivors.sort(key=lambda n: (n["pitch"], n["start"]))
    deduped: list[dict] = []
    last_by_pitch: dict[int, dict] = {}
    for n in survivors:
        prev = last_by_pitch.get(n["pitch"])
        if prev is not None and n["start"] - prev["start"] <= _DEDUPE_WINDOW:
            prev_score = (prev.get("velocity", 0), prev.get("end", 0) - prev["start"])
            this_score = (n.get("velocity", 0), n.get("end", 0) - n["start"])
            if this_score > prev_score:
                deduped[-1 - deduped[::-1].index(prev)] = n  # noqa: RUF015
                last_by_pitch[n["pitch"]] = n
            continue
        deduped.append(n)
        last_by_pitch[n["pitch"]] = n

    deduped.sort(key=lambda n: (n["start"], n["pitch"]))
    return deduped


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def _load_overrides(paths: VideoPaths) -> dict[int, list[dict]]:
    """Return {cluster_id: [{"note_index": int, "string": int, "fret": int}, ...]}."""
    if not paths.frets_overrides_json.exists():
        return {}
    try:
        data = json.loads(paths.frets_overrides_json.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[int, list[dict]] = {}
    for entry in data.get("overrides", []):
        out[entry["cluster_id"]] = entry["new_assignments"]
    return out


def _apply_overrides(notes: list[dict], overrides: dict[int, list[dict]]) -> list[dict]:
    """Return a copy of ``notes`` with the override assignments applied."""
    if not overrides:
        return notes
    out: list[dict] = []
    # Build per-note overrides by (cluster_id, note_index).
    by_pair: dict[tuple[int, int], dict] = {}
    for cluster_id, assigns in overrides.items():
        for a in assigns:
            by_pair[(cluster_id, a["note_index"])] = a

    # We don't have note_index *within the cluster* in the frets.json notes
    # records, so derive it from order within the cluster.
    cluster_local_idx: dict[int, int] = defaultdict(int)
    for n in notes:
        cid = n["cluster_id"]
        local = cluster_local_idx[cid]
        cluster_local_idx[cid] += 1
        key = (cid, local)
        if key in by_pair:
            replacement = dict(n)
            replacement["string"] = by_pair[key]["string"]
            replacement["fret"] = by_pair[key]["fret"]
            replacement["overridden"] = True
            out.append(replacement)
        else:
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Canonical instance selection
# ---------------------------------------------------------------------------


def _pick_canonical_instance(section: dict) -> dict | None:
    instances = section.get("instances", [])
    if not instances:
        return None

    def score(inst: dict) -> tuple[int, float]:
        quality_rank = _DEMO_QUALITY_RANK.get(inst.get("demo_quality"), 0)
        duration = inst["end"] - inst["start"]
        return (quality_rank, duration)

    return max(instances, key=score)


# ---------------------------------------------------------------------------
# ASCII tab rendering
# ---------------------------------------------------------------------------


def _render_section_tab(
    section_notes: list[dict],
    line_width: int,
    beat_times: list[float],
    string_letters: list[str] | None = None,
) -> tuple[str, int, list[str]]:
    """Render section's notes as a beat-aligned 6-line ASCII tab.

    The visible columns correspond to 8th-note subdivisions of the
    detected beat grid. Cluster onsets are snapped to the nearest
    subdivision. Beat downbeats are marked with a wider separator;
    every ``_BEATS_PER_BAR`` beats a bar line `|` is drawn.

    Returns ``(ascii_tab, slot_collisions, layout_notes)``: how many
    clusters snapped onto an already-occupied slot (most are resolved by
    shifting forward to the next free slot), plus footnotes for notes
    dropped by same-string conflicts and for distinct onsets that still
    share a cell cross-string after the forward scan.
    """
    if not section_notes:
        return "", 0, []

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for n in section_notes:
        by_cluster[n["cluster_id"]].append(n)
    cluster_ids = sorted(by_cluster, key=lambda cid: min(n["start"] for n in by_cluster[cid]))

    letters = string_letters or _DEFAULT_TAB_STRING_LETTERS

    grid = _subdivisions_from_beats(beat_times)
    if not grid:
        # Beat detection produced nothing usable — fall back to event-ordered.
        return _render_event_ordered(by_cluster, cluster_ids, line_width, letters), 0, []

    # Map each cluster to its nearest grid slot. Two clusters snapping onto
    # the same slot used to merge silently (same-string notes last-write-
    # won, so separate plucks fused into never-played chords). Now a
    # later-onset cluster is bumped forward to the NEXT FREE subdivision
    # slot within _MAX_SLOT_SHIFT. Clusters are placed in onset order and
    # the scan only moves forward, so bumped clusters can cascade (3rd
    # cluster lands 2 slots over) without ever inverting onset order.
    # When no slot in budget is free the clusters share the cell; per-
    # string conflicts and cross-string merges are footnoted below.
    grid_times = [t for t, _ in grid]
    section_t0 = grid_times[0]
    cluster_at_slot: dict[int, list[dict]] = defaultdict(list)
    slot_collisions = 0
    layout_notes: list[str] = []
    for cid in cluster_ids:
        t = min(n["start"] for n in by_cluster[cid])
        slot = _nearest_index(grid_times, t)
        if slot in cluster_at_slot:
            slot_collisions += 1
            limit = min(slot + _MAX_SLOT_SHIFT, len(grid_times) - 1)
            for candidate in range(slot + 1, limit + 1):
                if candidate not in cluster_at_slot:
                    slot = candidate
                    break
        cluster_at_slot[slot].extend(by_cluster[cid])

    # Build a cell per grid slot (most are empty rests).
    raw_cells: list[list[str]] = []
    slot_markers: list[str] = []  # "" / "beat" / "bar"
    beat_count = 0
    for slot_idx, (_, is_beat) in enumerate(grid):
        cell = ["-"] * 6
        kept_by_line: dict[int, dict] = {}
        for n in cluster_at_slot.get(slot_idx, []):
            line_idx = 5 - n["string"]
            prev = kept_by_line.get(line_idx)
            if prev is None:
                kept_by_line[line_idx] = n
                continue
            kept, dropped = _slot_winner(prev, n)
            kept_by_line[line_idx] = kept
            layout_notes.append(
                f"same-string conflict at +{dropped['start'] - section_t0:.1f}s — "
                f"string {letters[line_idx]}: kept fret {kept['fret']}, "
                f"dropped fret {dropped['fret']} (two notes on one string "
                "in the same grid slot)"
            )
        # Distinct clusters still sharing one cell after the forward scan
        # would be drawn as a single chord the player never strummed —
        # never let that happen silently.
        kept_clusters = {n["cluster_id"] for n in kept_by_line.values()}
        if len(kept_clusters) > 1:
            cell_t = min(n["start"] for n in kept_by_line.values())
            layout_notes.append(
                f"cross-string merge at +{cell_t - section_t0:.1f}s — "
                f"{len(kept_clusters)} separate onsets share one cell (no "
                f"free slot within {_MAX_SLOT_SHIFT} subdivisions); drawn "
                "as a single chord"
            )
        for line_idx, n in kept_by_line.items():
            cell[line_idx] = str(n["fret"])
        cell_width = max(len(c) for c in cell)
        cell = [c.rjust(cell_width, "-") for c in cell]
        raw_cells.append(cell)
        if is_beat:
            beat_count += 1
            slot_markers.append("bar" if beat_count % _BEATS_PER_BAR == 1 else "beat")
        else:
            slot_markers.append("")

    if len(layout_notes) > _MAX_LAYOUT_NOTES_PER_SECTION:
        extra = len(layout_notes) - _MAX_LAYOUT_NOTES_PER_SECTION
        layout_notes = layout_notes[:_MAX_LAYOUT_NOTES_PER_SECTION]
        layout_notes.append(f"… (+{extra} more layout conflicts)")
    tab = _pack_quantized(raw_cells, slot_markers, line_width, letters)
    return tab, slot_collisions, layout_notes


def _slot_winner(a: dict, b: dict) -> tuple[dict, dict]:
    """Pick which of two same-string, same-slot notes survives: the
    earlier onset wins; at equal onsets the longer note wins. Returns
    ``(kept, dropped)``."""

    def key(n: dict) -> tuple[float, float]:
        return (n["start"], -(n.get("end", n["start"]) - n["start"]))

    return (a, b) if key(a) <= key(b) else (b, a)


def _render_event_ordered(
    by_cluster: dict[int, list[dict]],
    cluster_ids: list[int],
    line_width: int,
    string_letters: list[str] | None = None,
) -> str:
    """Fallback when beat detection fails — old per-onset rendering."""
    raw_cells: list[list[str]] = []
    prev_end: float | None = None
    for cid in cluster_ids:
        cluster_notes = by_cluster[cid]
        cluster_start = min(n["start"] for n in cluster_notes)
        cluster_end = max(n.get("end", n["start"]) for n in cluster_notes)
        if prev_end is not None and cluster_start - prev_end > _PAUSE_GAP_SECONDS:
            raw_cells.append(_pause_cell())
        cell = ["-"] * 6
        for n in cluster_notes:
            line_idx = 5 - n["string"]
            cell[line_idx] = str(n["fret"])
        cell_width = max(len(c) for c in cell)
        cell = [c.rjust(cell_width, "-") for c in cell]
        raw_cells.append(cell)
        prev_end = cluster_end
    return _pack_into_systems(raw_cells, line_width, string_letters or _DEFAULT_TAB_STRING_LETTERS)


def _nearest_index(sorted_times: list[float], t: float) -> int:
    """Bisect to find the closest index in a sorted list of times."""
    import bisect

    i = bisect.bisect_left(sorted_times, t)
    if i == 0:
        return 0
    if i >= len(sorted_times):
        return len(sorted_times) - 1
    before = sorted_times[i - 1]
    after = sorted_times[i]
    return i - 1 if abs(t - before) <= abs(t - after) else i


def _pack_quantized(
    cells: list[list[str]],
    markers: list[str],
    line_width: int,
    string_letters: list[str],
) -> str:
    """Pack beat-quantized cells into systems, wrapping at bar boundaries.

    Each subdivision-cell is prefixed by `|` (when it starts a new bar) or
    `-` (otherwise). When a new bar boundary is reached and the in-progress
    line is already at or beyond the line-width budget, wrap there — the
    completed bars become a system and the new bar's first cell starts
    the next system. A closing `|` is appended to the very last system.
    """
    if not cells:
        return ""
    systems: list[list[str]] = []
    current = ["" for _ in range(6)]

    for idx, (cell, marker) in enumerate(zip(cells, markers, strict=True)):
        if idx == 0:
            prefix = ""
        elif marker == "bar":
            prefix = "|"
        else:
            prefix = "-"
        for i in range(6):
            current[i] += prefix + cell[i]
        # Wrap only when we just placed a `|` AND we're past the budget.
        if marker == "bar" and idx > 0 and len(current[0]) > line_width - 4:
            bar_end = len(current[0]) - len(cell[0])
            wrapped = [line[:bar_end] for line in current]
            leftover = [line[bar_end:] for line in current]
            systems.append(wrapped)
            current = leftover

    # Close out the final system with a trailing bar line.
    if current[0]:
        for i in range(6):
            current[i] += "|"
        systems.append(current)

    parts: list[str] = []
    for sys_lines in systems:
        block = []
        for letter, line in zip(string_letters, sys_lines, strict=True):
            block.append(f"  {letter:>2}|-{line}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def _pause_cell() -> list[str]:
    """A wider all-dash cell to mark a phrasing break."""
    return ["---"] * 6


def _pack_into_systems(cells: list[list[str]], line_width: int, string_letters: list[str]) -> str:
    systems: list[list[str]] = []
    current = ["" for _ in range(6)]
    for cell in cells:
        cell_width = len(cell[0])
        added = cell_width + 1  # cell + 1-char separator
        if current[0] and len(current[0]) + added > line_width - 4:  # 4 = " e|" prefix
            systems.append(current)
            current = ["" for _ in range(6)]
        for i in range(6):
            current[i] += cell[i] + "-"
    if current[0]:
        systems.append(current)

    parts: list[str] = []
    for sys_lines in systems:
        block = []
        for letter, line in zip(string_letters, sys_lines, strict=True):
            block.append(f"  {letter:>2}|-{line}|")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Document formatting
# ---------------------------------------------------------------------------


def _format_full_tab(
    sections_data: dict, rendered: list[RenderedSection], tuning: TuningInfo
) -> str:
    lines: list[str] = []
    video_id = sections_data.get("video_id", "?")
    lines.append(f"# migs-tab — video {video_id}")
    lines.append("")
    lines.append(f"Tuning: {tuning.label}")
    if tuning.capo:
        lines.append(f"Capo:   fret {tuning.capo}  (fret numbers below are RELATIVE to the capo)")
    lines.append(
        f"Strings (low → high): {tuning.strings_midi}  · detection source: {tuning.source}"
    )
    lines.append("")
    summary = sections_data.get("structural_summary")
    if summary:
        lines.append(summary)
        lines.append("")
    lines.append(
        f"Sections rendered: {len(rendered)}.  "
        "Section order follows sections.json (typically tutorial-teaching "
        "order, which is also a sensible play order for the song)."
    )
    lines.append("")
    for sec in rendered:
        collision_note = f", {sec.slot_collisions} slot collision(s)" if sec.slot_collisions else ""
        lines.append("=" * 72)
        lines.append(
            f"[{sec.label}]   {sec.canonical_start:.1f}-{sec.canonical_end:.1f}s "
            f"({sec.note_count} notes, {sec.cluster_count} clusters, "
            f"~{sec.tempo_bpm:.0f} bpm{collision_note})"
        )
        if sec.chord_progression:
            lines.append(f"chords: {'  '.join(sec.chord_progression[:16])}")
        if sec.description:
            # Wrap description lightly.
            lines.append("")
            for chunk in _word_wrap(sec.description, 72):
                lines.append(chunk)
        lines.append("")
        lines.append(sec.ascii_tab)
        if sec.layout_notes:
            lines.append("")
            lines.append("layout notes (beat-grid collisions):")
            for note in sec.layout_notes:
                lines.append(f"  • {note}")
        if sec.hints:
            lines.append("")
            lines.append("basic-pitch hints (ornaments MT3 simplified out):")
            for h in sec.hints:
                lines.append(f"  • {h}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _format_markdown(
    sections_data: dict, rendered: list[RenderedSection], tuning: TuningInfo
) -> str:
    parts: list[str] = []
    video_id = sections_data.get("video_id", "?")
    parts.append(f"# migs-tab — `{video_id}`\n")
    parts.append(f"**Tuning:** {tuning.label}")
    if tuning.capo:
        parts.append(
            f"  \n**Capo:** fret {tuning.capo} _(fret numbers in tabs are relative to the capo)_"
        )
    parts.append(
        f"  \nStrings low → high: `{tuning.strings_midi}` · detection source: `{tuning.source}`\n"
    )
    summary = sections_data.get("structural_summary")
    if summary:
        parts.append(f"_{summary}_\n")
    for sec in rendered:
        collision_note = (
            f" · {sec.slot_collisions} slot collision(s)" if sec.slot_collisions else ""
        )
        parts.append(f"## {sec.label}")
        parts.append(
            f"_{sec.canonical_start:.1f}-{sec.canonical_end:.1f}s · "
            f"{sec.note_count} notes · ~{sec.tempo_bpm:.0f} bpm{collision_note} · "
            f"chords: {' '.join(sec.chord_progression[:16])}_\n"
        )
        if sec.description:
            parts.append(sec.description + "\n")
        parts.append("```")
        parts.append(sec.ascii_tab)
        parts.append("```")
        if sec.layout_notes:
            parts.append("\n**layout notes** _(beat-grid collisions)_:\n")
            for note in sec.layout_notes:
                parts.append(f"- {note}")
            parts.append("")
        if sec.hints:
            parts.append("\n**basic-pitch hints** _(ornaments MT3 simplified out)_:\n")
            for h in sec.hints:
                parts.append(f"- {h}")
            parts.append("")
        parts.append("")
    return "\n".join(parts)


def _word_wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    line = ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out
