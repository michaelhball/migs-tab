"""Phase 4: render a section-by-section ASCII guitar tab.

Reads ``sections.json`` (which segments of the video correspond to which
parts of the song, grouped by repetition), picks the *canonical* instance
of each section (prefer slow-walkthroughs, then normal-tempo, then the
longest take), filters ``frets.json`` to that instance's time window,
applies any ``frets.overrides.json`` corrections from the vision pass,
and emits a 6-line ASCII tab grouped by section label.

The output isn't beat-quantized — basic-pitch onsets are jittery and we
don't have a reliable beat track yet — so the tab is *event-ordered*:
each onset cluster becomes one column, with extra spacing inserted where
the on-disk time gap exceeds a tunable threshold so the eye can see
phrasing breaks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .paths import DEFAULT_OUTPUT_DIR, VideoPaths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Conventional tab order: top line = high E (string 5 in our internal index),
# bottom line = low E (string 0 in our internal index).
_TAB_STRING_LETTERS = ["e", "B", "G", "D", "A", "E"]

# Wrap each tab system at this many characters.
_DEFAULT_LINE_WIDTH = 72

# If the gap between two consecutive onset clusters exceeds this many
# seconds, insert an extra "rest" column to suggest a pause to the reader.
_PAUSE_GAP_SECONDS = 0.8

# Sections whose chord_progression is empty are talking-only and skipped.
# Plus a hard exclusion list for sections we never want to render.
_SKIP_LABELS = {"closing_remarks"}

# Demo-quality ranking when picking a canonical instance.
_DEMO_QUALITY_RANK = {
    "slow-walkthrough": 3,
    "normal-tempo": 2,
    "repeated-loop": 2,
    "partial": 1,
}


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


def render(
    paths: VideoPaths,
    output_root: Path = DEFAULT_OUTPUT_DIR,
    line_width: int = _DEFAULT_LINE_WIDTH,
    force: bool = False,
) -> Path:
    """Render the section-by-section tab. Returns the tab.txt path."""
    if not paths.sections_json.exists():
        raise FileNotFoundError(
            f"sections.json not found at {paths.sections_json}; run the skill's "
            "Phase 2 step first to produce it."
        )
    if not paths.frets_json.exists():
        raise FileNotFoundError(
            f"frets.json not found at {paths.frets_json}; run `migs-tab frets` first."
        )

    out_dir = paths.output_dir(output_root)
    tab_path = out_dir / "tab.txt"
    md_path = out_dir / "tab.md"
    if tab_path.exists() and not force:
        return tab_path

    sections_data = json.loads(paths.sections_json.read_text())
    frets_data = json.loads(paths.frets_json.read_text())
    overrides = _load_overrides(paths)
    notes = _apply_overrides(frets_data["notes"], overrides)

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
        ascii_tab = _render_section_tab(section_notes, line_width=line_width)
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
            )
        )

    tab_text = _format_full_tab(sections_data, rendered)
    tab_path.write_text(tab_text)
    md_path.write_text(_format_markdown(sections_data, rendered))
    return tab_path


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


def _render_section_tab(section_notes: list[dict], line_width: int) -> str:
    """Render section's notes as a wrapped 6-line ASCII tab."""
    # Group notes into onset clusters (cluster_id from frets.json).
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for n in section_notes:
        by_cluster[n["cluster_id"]].append(n)
    cluster_ids = sorted(by_cluster, key=lambda cid: min(n["start"] for n in by_cluster[cid]))

    if not cluster_ids:
        return ""

    # Build raw columns. Each column is a tuple (cell_strings, separator_width).
    raw_cells: list[list[str]] = []
    prev_end: float | None = None
    for cid in cluster_ids:
        cluster_notes = by_cluster[cid]
        cluster_start = min(n["start"] for n in cluster_notes)
        cluster_end = max(n.get("end", n["start"]) for n in cluster_notes)

        # Insert a "pause" cell if the gap to the previous cluster is large.
        if prev_end is not None and cluster_start - prev_end > _PAUSE_GAP_SECONDS:
            raw_cells.append(_pause_cell())

        cell = ["-"] * 6
        for n in cluster_notes:
            line_idx = 5 - n["string"]  # tab top line = high E (string 5)
            cell[line_idx] = str(n["fret"])
        # Pad each line in the cell to the same width.
        cell_width = max(len(c) for c in cell)
        cell = [c.rjust(cell_width, "-") for c in cell]
        raw_cells.append(cell)
        prev_end = cluster_end

    # Pack cells into systems of width ~ line_width.
    return _pack_into_systems(raw_cells, line_width)


def _pause_cell() -> list[str]:
    """A wider all-dash cell to mark a phrasing break."""
    return ["---"] * 6


def _pack_into_systems(cells: list[list[str]], line_width: int) -> str:
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
        for letter, line in zip(_TAB_STRING_LETTERS, sys_lines, strict=True):
            block.append(f"  {letter}|-{line}|")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Document formatting
# ---------------------------------------------------------------------------


def _format_full_tab(sections_data: dict, rendered: list[RenderedSection]) -> str:
    lines: list[str] = []
    video_id = sections_data.get("video_id", "?")
    lines.append(f"# migs-tab — video {video_id}")
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
            f"({sec.note_count} notes, {sec.cluster_count} clusters)"
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


def _format_markdown(sections_data: dict, rendered: list[RenderedSection]) -> str:
    parts: list[str] = []
    video_id = sections_data.get("video_id", "?")
    parts.append(f"# migs-tab — `{video_id}`\n")
    summary = sections_data.get("structural_summary")
    if summary:
        parts.append(f"_{summary}_\n")
    for sec in rendered:
        parts.append(f"## {sec.label}")
        parts.append(
            f"_{sec.canonical_start:.1f}-{sec.canonical_end:.1f}s · "
            f"{sec.note_count} notes · chords: {' '.join(sec.chord_progression[:16])}_\n"
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
