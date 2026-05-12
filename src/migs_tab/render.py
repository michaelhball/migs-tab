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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

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
_DEDUPE_WINDOW = 0.10  # drop duplicate same-pitch notes within this many seconds

# Sustain detection: a note whose onset falls inside (a slightly contracted
# version of) an earlier same-pitch note's duration is treated as a
# re-detection of that ringing string rather than a fresh pluck.
_SUSTAIN_CONTRACTION = 0.05  # subtract this from the earlier note's end

# Cross-instance voting: a canonical-instance note is kept only if its
# (chord_name, pitch_class) pair appears in at least this many *other*
# instances of the same section. 1 = "at least one other take confirms".
# Skipped entirely for sections with only one instance.
_CROSS_INSTANCE_MIN_SUPPORT = 1

# Demo-quality ranking when picking a canonical instance.
_DEMO_QUALITY_RANK = {
    "slow-walkthrough": 3,
    "normal-tempo": 2,
    "repeated-loop": 2,
    "partial": 1,
}

# Beat-tracking + quantization parameters.
_BEAT_SR = 22050
_SUBDIVISIONS_PER_BEAT = 2  # 8th notes; bump to 4 for 16th-note quantization
_BEATS_PER_BAR = 4  # most acoustic guitar tutorials are in 4/4
# Fallback tempo to use when beat tracking fails / yields no beats.
_FALLBACK_TEMPO_BPM = 90.0


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
    if tab_path.exists() and not force:
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

    # Build cross-instance pitch-class support per section, so we can drop
    # canonical-instance notes that no other take confirms.
    chord_spans = _load_chord_spans_for_render(paths)
    cross_support = _build_cross_instance_support(sections_data, notes, chord_spans)

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
        ascii_tab = _render_section_tab(
            section_notes,
            line_width=line_width,
            beat_times=beat_times,
            string_letters=tuning_info.string_letters,
        )
        clusters_in_section = {n["cluster_id"] for n in section_notes}
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
    return tab_path


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
    ``cross_support_min + 1`` instances. Loud notes (velocity ≥ 75)
    survive even without cross-instance backup — the canonical's
    confidence on its own is decisive."""
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
        elif n.get("velocity", 999) >= 75:
            survivors.append(n)
    return survivors


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------


def _detect_beats(audio_path: Path, start_s: float, end_s: float) -> tuple[float, list[float]]:
    """Run librosa beat tracking on the [start, end] slice of the guitar stem.

    Returns (tempo_bpm, absolute beat times in the video's time frame).
    Falls back to a uniform grid at ``_FALLBACK_TEMPO_BPM`` if librosa
    finds no beats (very rare on guitar-stem audio).
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
    return tempo_val, extended


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
    """Drop short, quiet, duplicate, or sustained-continuation notes."""
    survivors: list[dict] = []
    for n in notes:
        dur = n.get("end", n["start"]) - n["start"]
        if dur < _MIN_NOTE_DURATION:
            continue
        vel = n.get("velocity")
        if vel is not None and vel < _MIN_NOTE_VELOCITY:
            continue
        survivors.append(n)

    # Dedupe close-onset same-pitch detections, prefer the louder/longer.
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

    # Sustain detection: scan in time order. For each note, look at the most
    # recent same-pitch note that we kept; if THIS note's onset falls inside
    # (prev.start, prev.end - _SUSTAIN_CONTRACTION), the new "onset" is most
    # likely a re-detection of the still-ringing string, so drop it.
    deduped.sort(key=lambda n: (n["start"], n["pitch"]))
    out: list[dict] = []
    last_kept_by_pitch: dict[int, dict] = {}
    for n in deduped:
        prev = last_kept_by_pitch.get(n["pitch"])
        if prev is not None:
            prev_end_contracted = prev.get("end", prev["start"]) - _SUSTAIN_CONTRACTION
            if prev["start"] < n["start"] < prev_end_contracted:
                continue
        out.append(n)
        last_kept_by_pitch[n["pitch"]] = n
    return out


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
) -> str:
    """Render section's notes as a beat-aligned 6-line ASCII tab.

    The visible columns correspond to 8th-note subdivisions of the
    detected beat grid. Cluster onsets are snapped to the nearest
    subdivision. Beat downbeats are marked with a wider separator;
    every ``_BEATS_PER_BAR`` beats a bar line `|` is drawn.
    """
    if not section_notes:
        return ""

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for n in section_notes:
        by_cluster[n["cluster_id"]].append(n)
    cluster_ids = sorted(by_cluster, key=lambda cid: min(n["start"] for n in by_cluster[cid]))

    letters = string_letters or _DEFAULT_TAB_STRING_LETTERS

    grid = _subdivisions_from_beats(beat_times)
    if not grid:
        # Beat detection produced nothing usable — fall back to event-ordered.
        return _render_event_ordered(by_cluster, cluster_ids, line_width, letters)

    # Map each cluster to its nearest grid slot.
    grid_times = [t for t, _ in grid]
    cluster_at_slot: dict[int, list[dict]] = defaultdict(list)
    for cid in cluster_ids:
        t = min(n["start"] for n in by_cluster[cid])
        slot = _nearest_index(grid_times, t)
        cluster_at_slot[slot].extend(by_cluster[cid])

    # Build a cell per grid slot (most are empty rests).
    raw_cells: list[list[str]] = []
    slot_markers: list[str] = []  # "" / "beat" / "bar"
    beat_count = 0
    for slot_idx, (_, is_beat) in enumerate(grid):
        cell = ["-"] * 6
        notes_here = cluster_at_slot.get(slot_idx, [])
        for n in notes_here:
            line_idx = 5 - n["string"]
            cell[line_idx] = str(n["fret"])
        cell_width = max(len(c) for c in cell)
        cell = [c.rjust(cell_width, "-") for c in cell]
        raw_cells.append(cell)
        if is_beat:
            beat_count += 1
            slot_markers.append("bar" if beat_count % _BEATS_PER_BAR == 1 else "beat")
        else:
            slot_markers.append("")

    return _pack_quantized(raw_cells, slot_markers, line_width, letters)


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
        lines.append("=" * 72)
        lines.append(
            f"[{sec.label}]   {sec.canonical_start:.1f}-{sec.canonical_end:.1f}s "
            f"({sec.note_count} notes, {sec.cluster_count} clusters, "
            f"~{sec.tempo_bpm:.0f} bpm)"
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
        parts.append(f"## {sec.label}")
        parts.append(
            f"_{sec.canonical_start:.1f}-{sec.canonical_end:.1f}s · "
            f"{sec.note_count} notes · ~{sec.tempo_bpm:.0f} bpm · "
            f"chords: {' '.join(sec.chord_progression[:16])}_\n"
        )
        if sec.description:
            parts.append(sec.description + "\n")
        parts.append("```")
        parts.append(sec.ascii_tab)
        parts.append("```\n")
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
