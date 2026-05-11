"""Phase 3: assign string/fret positions to a sequence of detected notes.

We group notes into *onset clusters* (notes whose onsets fall within a small
window — these get played together as a chord or rapid arpeggio under one
hand position), then find the optimal sequence of hand positions across all
clusters via Viterbi dynamic programming.

For each cluster we enumerate hand-shapes (assignments of one string per
note, with all frets within a reachable hand-span). Each shape has an
intra-cluster cost (penalizes wide spans, rewards open strings, penalizes
very high positions) and a transition cost to the previous cluster's shape
(penalizes large changes in hand position).

After the heuristic pass we flag *ambiguous* clusters — those whose best
shape was only marginally better than the runner-up — for the LLM vision
pass (Phase 3.5) to disambiguate from frame imagery.

The output ``frets.json`` records, for every note, the chosen (string, fret)
plus an ``ambiguous`` flag, an ``alternatives`` list (other near-optimal
shapes for the cluster), and the cluster ID.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product

from .paths import VideoPaths

# ---------------------------------------------------------------------------
# Constants — standard tuning (low → high), reasonable fret/hand limits
# ---------------------------------------------------------------------------

# Low E (string 6 traditionally) → high E (string 1 traditionally).
# We index 0..5 where 0 = low E, 5 = high E; the printed tab string number
# (the conventional 6..1) is computed as 6 - i.
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)
NUM_STRINGS = 6
MAX_FRET = 19  # most acoustic guitars have 19-22 accessible frets

# Maximum reach a player can comfortably make with index → pinky.
# 4 fret reach is comfortable; 5 is a stretch; 6+ is rare. We allow up to 5.
MAX_HAND_SPAN = 5

# Onset cluster window — notes whose onsets fall within this many seconds of
# each other are considered "played together" (a chord stab, a pinch, the
# start of an arpeggio that all comes from one hand position). 0.06s was too
# tight — basic-pitch jitters chord onsets by 20-50ms and a strummed chord
# often spans ~100ms. 0.15s captures strummed chords while still keeping
# arpeggio notes in distinct clusters.
ONSET_CLUSTER_SECONDS = 0.15

# Weights for the cluster's intrinsic cost (lower = better).
# Tuned so an open-position shape scores strongly negative (driving the
# algorithm toward cowboy chords by default) and the same notes up at fret
# 7-12 score notably higher. Hand-movement is intentionally light: tutorials
# do move the hand around, so penalizing motion too hard locks the
# algorithm into one position for the whole song.
_W_FRET_POSITION = 0.15  # average fret position (strong pull to low frets)
_W_FRET_SPAN = 0.6  # max-min fret span (encourage compact shapes)
_W_OPEN_BONUS = -0.8  # per open string in the shape (strong reward)
_W_HIGH_FRET_PENALTY = 0.4  # extra penalty above fret 12

# Weights for the transition cost between consecutive clusters.
# Lighter than the initial 0.35: the chord prior now does most of the
# anchoring, and hand-move shouldn't be strong enough to trap the path
# in a high-fret position when the next cluster has an obvious open-chord
# match.
_W_HAND_MOVE = 0.15
_W_STRING_PRESERVE = -0.15  # bonus per note still on the same string

# An assignment is "ambiguous" if the runner-up shape was within this delta.
# 0.4 was way too generous and flagged ~70% of clusters; 0.12 picks out
# only the genuinely close calls (a single fret/string disagreement at
# similar position cost).
_AMBIGUITY_MARGIN = 0.12

# Idiomatic chord-shape bonus: when a cluster's pitches form a subset of a
# known open-position chord template AND the candidate shape's (string, fret)
# choices exactly match the template's voicing for those pitches, apply this
# negative cost. Scaled by coverage (full chord = full bonus, 1-of-5 = 20%).
# Large enough to overcome ~5-7 frets of hand movement at the current
# _W_HAND_MOVE, so the algorithm will jump back to an open chord shape
# even when it's currently anchored on a high-fret passage.
_CHORD_SHAPE_BONUS = -4.0

# Library of idiomatic open-position chord voicings keyed by chord name.
# Each template maps pitch (MIDI) → (string_index, fret) under standard
# tuning (low E = string 0). These are the cowboy chord shapes a guitarist
# defaults to when notes are in range; the algorithm should strongly
# prefer these voicings over arbitrary high-fret alternatives.
#
# String index reminder: 0=low E (E2 open), 1=A, 2=D, 3=G, 4=B, 5=high E (E4 open).
_CHORD_TEMPLATES: dict[str, dict[int, tuple[int, int]]] = {
    # E minor — open
    "Em": {40: (0, 0), 47: (1, 2), 52: (2, 2), 55: (3, 0), 59: (4, 0), 64: (5, 0)},
    # E major — open
    "E": {40: (0, 0), 47: (1, 2), 52: (2, 2), 56: (3, 1), 59: (4, 0), 64: (5, 0)},
    # E7 — open (D string open in middle)
    "E7": {40: (0, 0), 47: (1, 2), 50: (2, 0), 56: (3, 1), 59: (4, 0), 64: (5, 0)},
    # A minor — open
    "Am": {45: (1, 0), 52: (2, 2), 57: (3, 2), 60: (4, 1), 64: (5, 0)},
    # A major — open
    "A": {45: (1, 0), 52: (2, 2), 57: (3, 2), 61: (4, 2), 64: (5, 0)},
    # A7 — open (D fret 2, G open, B fret 2, high E open)
    "A7": {45: (1, 0), 52: (2, 2), 55: (3, 0), 61: (4, 2), 64: (5, 0)},
    # D major — open
    "D": {50: (2, 0), 57: (3, 2), 62: (4, 3), 66: (5, 2)},
    # D minor — open
    "Dm": {50: (2, 0), 57: (3, 2), 62: (4, 3), 65: (5, 1)},
    # D7 — open
    "D7": {50: (2, 0), 57: (3, 2), 60: (4, 1), 66: (5, 2)},
    # G major — open (the classic 3-2-0-0-0-3 voicing)
    "G": {43: (0, 3), 47: (1, 2), 50: (2, 0), 55: (3, 0), 59: (4, 0), 67: (5, 3)},
    # G major — alternative (3-2-0-0-3-3 with high B fret 3)
    "G_alt": {43: (0, 3), 47: (1, 2), 50: (2, 0), 55: (3, 0), 62: (4, 3), 67: (5, 3)},
    # C major — open
    "C": {48: (1, 3), 52: (2, 2), 55: (3, 0), 60: (4, 1), 64: (5, 0)},
    # F major — small "partial F" (top four strings): D fret 3, G fret 2, B fret 1, high E fret 1
    "F_partial": {53: (2, 3), 57: (3, 2), 60: (4, 1), 65: (5, 1)},
    # B minor — barre at fret 2 (common in folk/rock arrangements)
    "Bm": {47: (1, 2), 54: (2, 4), 59: (3, 4), 62: (4, 3), 66: (5, 2)},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FretAssign:
    """One note's chosen (string, fret) plus the index of the note within
    the original notes list."""

    note_index: int
    string: int  # 0 = low E, 5 = high E
    fret: int


@dataclass
class Shape:
    """A complete hand-position assignment for one onset cluster.

    Each entry is (note_index_within_cluster, string, fret). The list is
    sorted by note_index_within_cluster for stable comparison.
    """

    assignments: tuple[tuple[int, int, int], ...]
    cost: float

    def fret_positions(self) -> list[int]:
        # Open strings (fret 0) don't really count toward hand position.
        return [f for _, _, f in self.assignments if f > 0]

    def avg_fret_position(self) -> float:
        positions = self.fret_positions()
        if not positions:
            return 0.0
        return sum(positions) / len(positions)

    def string_to_fret_map(self) -> dict[int, int]:
        return {s: f for _, s, f in self.assignments}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_frets(paths: VideoPaths, force: bool = False) -> VideoPaths:
    """Run Viterbi fret assignment over the full notes.json, write frets.json."""
    if paths.frets_json.exists() and not force:
        return paths

    if not paths.notes_json.exists():
        raise FileNotFoundError(
            f"notes.json not found at {paths.notes_json}; run transcribe first."
        )

    notes = json.loads(paths.notes_json.read_text())["notes"]
    if not notes:
        paths.frets_json.write_text(json.dumps({"note_count": 0, "notes": []}, indent=2))
        return paths

    notes = _dedupe_same_pitch_onsets(notes)
    clusters = _cluster_notes_by_onset(notes)
    cluster_shapes = [_enumerate_shapes(notes, c) for c in clusters]

    # Drop clusters where no playable shape exists — these are pitches
    # outside guitar range or impossible-to-finger combinations. Annotate them.
    playable: list[tuple[list[int], list[Shape]]] = []
    unplayable_clusters: list[dict] = []
    for cidx, (cluster, shapes) in enumerate(zip(clusters, cluster_shapes, strict=True)):
        if shapes:
            playable.append((cluster, shapes))
        else:
            unplayable_clusters.append(
                {
                    "cluster_id": cidx,
                    "note_indices": cluster,
                    "pitches": [notes[i]["pitch"] for i in cluster],
                    "reason": "No reachable hand-shape (out-of-range pitch or impossible span).",
                }
            )

    chosen = _viterbi(playable)
    ambiguous_flags = _flag_ambiguous(playable, chosen)

    # Build per-note output.
    note_records: list[dict] = []
    cluster_id_of_note: dict[int, int] = {}
    cluster_records: list[dict] = []
    for cidx_out, ((cluster, shapes), shape, is_ambig) in enumerate(
        zip(playable, chosen, ambiguous_flags, strict=True)
    ):
        assignment_by_note = {ni: (s, f) for ni, s, f in shape.assignments}
        for note_local_idx, note_global_idx in enumerate(cluster):
            cluster_id_of_note[note_global_idx] = cidx_out
            s, f = assignment_by_note[note_local_idx]
            n = notes[note_global_idx]
            note_records.append(
                {
                    "note_index": note_global_idx,
                    "start": n["start"],
                    "end": n["end"],
                    "pitch": n["pitch"],
                    "string": s,  # 0..5, 0 = low E
                    "tab_string": NUM_STRINGS - s,  # 1..6, 1 = low E? — no, 6 = low E
                    "fret": f,
                    "cluster_id": cidx_out,
                    "ambiguous": is_ambig,
                }
            )
        cluster_records.append(
            {
                "cluster_id": cidx_out,
                "onset": min(notes[i]["start"] for i in cluster),
                "note_indices": cluster,
                "best_shape_cost": shape.cost,
                "ambiguous": is_ambig,
                "alternatives": _alternative_shapes(shapes, shape),
            }
        )

    out = {
        "note_count": len(note_records),
        "cluster_count": len(cluster_records),
        "ambiguous_cluster_count": sum(1 for c in cluster_records if c["ambiguous"]),
        "unplayable_clusters": unplayable_clusters,
        "tuning": {"low_to_high_midi": list(STANDARD_TUNING)},
        "params": {
            "max_fret": MAX_FRET,
            "max_hand_span": MAX_HAND_SPAN,
            "onset_cluster_seconds": ONSET_CLUSTER_SECONDS,
            "ambiguity_margin": _AMBIGUITY_MARGIN,
        },
        "clusters": cluster_records,
        "notes": note_records,
    }
    paths.frets_json.write_text(json.dumps(out, indent=2))
    return paths


# ---------------------------------------------------------------------------
# Pre-clustering: dedupe near-duplicate same-pitch onsets
# ---------------------------------------------------------------------------

# Window (seconds) within which two same-pitch detections are treated as a
# single note. basic-pitch occasionally splits one struck note into two
# overlapping detections; including both in a cluster forces them onto
# different strings and breaks chord-template matching.
_SAME_PITCH_DEDUPE_WINDOW = 0.20


def _dedupe_same_pitch_onsets(notes: list[dict]) -> list[dict]:
    """Drop duplicate onsets where another note of the same pitch starts within
    ``_SAME_PITCH_DEDUPE_WINDOW`` and the surviving copy is louder/longer."""
    by_pitch_sorted = sorted(notes, key=lambda n: (n["pitch"], n["start"]))
    survivors: list[dict] = []
    last_by_pitch: dict[int, dict] = {}
    for n in by_pitch_sorted:
        prev = last_by_pitch.get(n["pitch"])
        if prev is not None and n["start"] - prev["start"] <= _SAME_PITCH_DEDUPE_WINDOW:
            prev_score = (prev.get("velocity", 0), prev["end"] - prev["start"])
            this_score = (n.get("velocity", 0), n["end"] - n["start"])
            if this_score > prev_score:
                # Replace the previously-kept copy with this stronger one.
                for i, existing in enumerate(survivors):
                    if existing is prev:
                        survivors[i] = n
                        break
                last_by_pitch[n["pitch"]] = n
            continue
        survivors.append(n)
        last_by_pitch[n["pitch"]] = n
    survivors.sort(key=lambda n: (n["start"], n["pitch"]))
    return survivors


# ---------------------------------------------------------------------------
# Cluster notes by onset
# ---------------------------------------------------------------------------


def _cluster_notes_by_onset(notes: list[dict]) -> list[list[int]]:
    """Return a list of clusters, each a list of note indices.

    Two notes are in the same cluster iff their onsets are within
    ``ONSET_CLUSTER_SECONDS`` AND they're on different pitches with
    a reasonable hand-shape (we enforce the latter at enumerate time).
    """
    indexed = sorted(range(len(notes)), key=lambda i: notes[i]["start"])
    clusters: list[list[int]] = []
    current: list[int] = []
    current_anchor: float | None = None
    for i in indexed:
        s = notes[i]["start"]
        if current_anchor is None or s - current_anchor <= ONSET_CLUSTER_SECONDS:
            current.append(i)
            if current_anchor is None:
                current_anchor = s
        else:
            clusters.append(current)
            current = [i]
            current_anchor = s
    if current:
        clusters.append(current)
    return clusters


# ---------------------------------------------------------------------------
# Enumerate playable hand-shapes for one cluster
# ---------------------------------------------------------------------------


def _enumerate_shapes(notes: list[dict], cluster: list[int]) -> list[Shape]:
    """Enumerate all playable (string, fret) assignments for one cluster.

    Each note must go on a distinct string; the fretted frets (excluding
    open notes at fret 0) must fit within MAX_HAND_SPAN.
    """
    if not cluster:
        return []

    # For each note in the cluster, list its (string, fret) candidates.
    per_note_options: list[list[tuple[int, int]]] = []
    for nidx in cluster:
        pitch = notes[nidx]["pitch"]
        opts: list[tuple[int, int]] = []
        for s in range(NUM_STRINGS):
            fret = pitch - STANDARD_TUNING[s]
            if 0 <= fret <= MAX_FRET:
                opts.append((s, fret))
        per_note_options.append(opts)

    if any(not o for o in per_note_options):
        return []

    pitches = [notes[i]["pitch"] for i in cluster]
    pitch_set = frozenset(pitches)

    shapes: list[Shape] = []
    # Cartesian product of options. Bounded since cluster sizes are small
    # (typically 1-6 notes for a chord).
    for combo in product(*per_note_options):
        strings_used = [s for s, _ in combo]
        if len(set(strings_used)) != len(strings_used):
            continue  # same string assigned twice — not playable
        fretted = [f for _, f in combo if f > 0]
        if fretted:
            span = max(fretted) - min(fretted)
            if span > MAX_HAND_SPAN:
                continue
        assignments = tuple((i, s, f) for i, (s, f) in enumerate(combo))
        cost = _intrinsic_cost(combo) + _chord_shape_bonus(pitches, combo, pitch_set)
        shapes.append(Shape(assignments, cost))

    # Limit to the K best shapes per cluster to keep Viterbi tractable.
    shapes.sort(key=lambda x: x.cost)
    return shapes[:32]


def _chord_shape_bonus(
    pitches: list[int],
    combo: tuple[tuple[int, int], ...],
    pitch_set: frozenset[int],
) -> float:
    """Negative-cost bonus if this shape exactly matches a known chord
    template's voicing for the cluster's pitches.

    A template "applies" when the cluster's pitches are a subset of the
    template's notes AND the candidate shape places every note on the
    template-specified string/fret. Scaled by coverage so a full Am
    cluster gets the full bonus and a 2-of-5 partial gets ~40%.
    """
    if len(pitches) == 0:
        return 0.0
    best_bonus = 0.0
    for template in _CHORD_TEMPLATES.values():
        if not pitch_set <= template.keys():
            continue
        # Does this combo place every pitch where the template wants it?
        matches = all(template[pitches[i]] == (s, f) for i, (s, f) in enumerate(combo))
        if not matches:
            continue
        coverage = len(pitch_set) / len(template)
        bonus = _CHORD_SHAPE_BONUS * coverage
        if bonus < best_bonus:
            best_bonus = bonus
    return best_bonus


def _intrinsic_cost(combo: tuple[tuple[int, int], ...]) -> float:
    """Cost intrinsic to one hand-shape, independent of neighbors."""
    fretted = [f for _, f in combo if f > 0]
    open_count = sum(1 for _, f in combo if f == 0)
    if fretted:
        avg = sum(fretted) / len(fretted)
        span = max(fretted) - min(fretted)
        high_excess = max(0, max(fretted) - 12)
    else:
        avg = 0.0
        span = 0
        high_excess = 0
    return (
        _W_FRET_POSITION * avg
        + _W_FRET_SPAN * span
        + _W_OPEN_BONUS * open_count
        + _W_HIGH_FRET_PENALTY * high_excess
    )


def _transition_cost(prev: Shape, curr: Shape) -> float:
    """Cost to move from one shape's hand position to the next."""
    prev_avg = prev.avg_fret_position()
    curr_avg = curr.avg_fret_position()
    hand_move = abs(curr_avg - prev_avg)

    prev_strings = {s: f for _, s, f in prev.assignments}
    curr_strings = {s: f for _, s, f in curr.assignments}
    preserved = sum(
        1
        for s in curr_strings
        if s in prev_strings and curr_strings[s] == prev_strings[s] and curr_strings[s] > 0
    )

    return _W_HAND_MOVE * hand_move + _W_STRING_PRESERVE * preserved


# ---------------------------------------------------------------------------
# Viterbi over clusters
# ---------------------------------------------------------------------------


def _viterbi(playable: list[tuple[list[int], list[Shape]]]) -> list[Shape]:
    """Choose one Shape per cluster minimizing total cost."""
    if not playable:
        return []

    # dp[i][k] = best total cost to reach cluster i using shape k.
    # back[i][k] = which shape in cluster i-1 was the predecessor.
    n = len(playable)
    dp: list[list[float]] = []
    back: list[list[int]] = []

    init_shapes = playable[0][1]
    dp.append([sh.cost for sh in init_shapes])
    back.append([-1] * len(init_shapes))

    for i in range(1, n):
        shapes = playable[i][1]
        prev_shapes = playable[i - 1][1]
        row = []
        bt = []
        for sh in shapes:
            best_cost = float("inf")
            best_prev = 0
            for j, psh in enumerate(prev_shapes):
                c = dp[i - 1][j] + _transition_cost(psh, sh) + sh.cost
                if c < best_cost:
                    best_cost = c
                    best_prev = j
            row.append(best_cost)
            bt.append(best_prev)
        dp.append(row)
        back.append(bt)

    # Trace back.
    out: list[Shape] = [None] * n  # type: ignore[list-item]
    last_idx = min(range(len(dp[-1])), key=lambda k: dp[-1][k])
    out[-1] = playable[-1][1][last_idx]
    for i in range(n - 1, 0, -1):
        last_idx = back[i][last_idx]
        out[i - 1] = playable[i - 1][1][last_idx]
    return out


# ---------------------------------------------------------------------------
# Ambiguity detection + alternatives
# ---------------------------------------------------------------------------


def _flag_ambiguous(
    playable: list[tuple[list[int], list[Shape]]],
    chosen: list[Shape],
) -> list[bool]:
    """A cluster is ambiguous if any non-chosen shape sits within margin of
    the chosen shape's intrinsic cost. (We use intrinsic cost here rather
    than full Viterbi cost because the latter includes transitions which
    are properties of the *path*, not of the cluster decision itself.)"""
    flags: list[bool] = []
    for (_, shapes), pick in zip(playable, chosen, strict=True):
        # Find the second-best shape that differs from the chosen one in some
        # (string, fret) assignment (i.e., is a genuinely different choice).
        chosen_assigns = frozenset((s, f) for _, s, f in pick.assignments)
        diff_costs = [
            sh.cost
            for sh in shapes
            if frozenset((s, f) for _, s, f in sh.assignments) != chosen_assigns
        ]
        if not diff_costs:
            flags.append(False)
            continue
        runner_up = min(diff_costs)
        flags.append((runner_up - pick.cost) <= _AMBIGUITY_MARGIN)
    return flags


def _alternative_shapes(shapes: list[Shape], chosen: Shape, k: int = 3) -> list[dict]:
    """Return the top-k alternative shapes (excluding the chosen one)."""
    chosen_assigns = frozenset((s, f) for _, s, f in chosen.assignments)
    alts = [
        sh for sh in shapes if frozenset((s, f) for _, s, f in sh.assignments) != chosen_assigns
    ]
    out = []
    for sh in alts[:k]:
        out.append(
            {
                "cost": round(sh.cost, 4),
                "delta_vs_chosen": round(sh.cost - chosen.cost, 4),
                "assignments": [{"string": s, "fret": f} for _, s, f in sh.assignments],
            }
        )
    return out
