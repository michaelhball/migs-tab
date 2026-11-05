"""Score a video's fret assignment against hand-written reference passages.

Nothing in the pipeline can be judged without a reference, and the audio
self-consistency checks in ``verify.py`` cannot see an error that both
transcribers share. ``eval/<video_id>.json`` holds short passages (10-20 s)
whose correct fingering is known, usually because the instructor states it in
the captions. Each passage gives:

- ``vocabulary``: the (string, fret) positions the passage uses, as tab
  tokens like ``"A2"`` (A string, fret 2) or ``"e0"``. Every note the pipeline
  puts in the window is either in the vocabulary (right position) or not.
- ``sequence`` (optional): an ASCII tab of the passage in playing order, for
  riffs where the order is known. It is aligned to the pipeline's note
  clusters with dynamic programming and scored as note-level F1.

Metrics per passage:

- ``position_precision``: fraction of the pipeline's notes whose (string, fret)
  is in the vocabulary.
- ``pitch_precision``: fraction whose pitch is in the vocabulary's pitch set
  (separates transcription errors from string-choice errors).
- ``vocab_recall``: fraction of vocabulary positions that occur at least once.
- ``high_fret_strays``: notes above the vocabulary's highest fret + 2.
- ``sequence_f1`` (when a sequence is given): aligned note F1 on (string, fret).

Run with ``migs-tab eval <video_id>`` or ``migs-tab eval --all``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PROJECT_ROOT, VideoPaths

EVAL_DIR = PROJECT_ROOT / "eval"
_STRING_LETTERS = {"E": 0, "A": 1, "D": 2, "G": 3, "B": 4, "e": 5}
_STANDARD_OPEN = (40, 45, 50, 55, 59, 64)
_CLUSTER_GAP_S = 0.06
_TOKEN_RE = re.compile(r"^([EADGBe])(\d{1,2})$")


@dataclass
class Passage:
    label: str
    start: float
    end: float
    vocabulary: set[tuple[int, int]]
    sequence: list[set[tuple[int, int]]] | None
    source: str = ""
    confidence: str = ""


@dataclass
class PassageResult:
    label: str
    note_count: int
    position_precision: float | None
    pitch_precision: float | None
    vocab_recall: float | None
    high_fret_strays: int
    sequence_f1: float | None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


def parse_token(token: str) -> tuple[int, int]:
    m = _TOKEN_RE.match(token.strip())
    if not m:
        raise ValueError(f"bad position token {token!r}; expected like 'A2' or 'e0'")
    return _STRING_LETTERS[m.group(1)], int(m.group(2))


def parse_ascii_tab(lines: list[str]) -> list[set[tuple[int, int]]]:
    """Columns of an ASCII tab as ordered sets of (string, fret).

    Lines look like ``"A|-0h3---3-|"``. Only digits count; articulation
    letters, dashes and bar lines are ignored. Digits starting in the same
    column across strings form one chord; a two-digit fret occupies two
    columns and is read from its first."""
    rows: dict[int, str] = {}
    for line in lines:
        s = line.strip()
        if len(s) < 2 or s[0] not in _STRING_LETTERS or s[1] != "|":
            continue
        rows[_STRING_LETTERS[s[0]]] = s[2:]
    if not rows:
        return []
    width = max(len(r) for r in rows.values())
    events: dict[int, set[tuple[int, int]]] = {}
    for string, row in rows.items():
        col = 0
        while col < width:
            ch = row[col] if col < len(row) else "-"
            if ch.isdigit():
                end = col + 1
                if end < len(row) and row[end].isdigit():
                    end += 1
                events.setdefault(col, set()).add((string, int(row[col:end])))
                col = end
            else:
                col += 1
    return [events[c] for c in sorted(events)]


def load_passages(video_id: str, eval_dir: Path = EVAL_DIR) -> list[Passage]:
    path = eval_dir / f"{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no reference passages at {path}")
    data = json.loads(path.read_text())
    out: list[Passage] = []
    for p in data["passages"]:
        vocab = {parse_token(t) for t in p["vocabulary"]}
        seq = parse_ascii_tab(p["sequence"]) if p.get("sequence") else None
        if seq:
            vocab |= {pos for chord in seq for pos in chord}
        out.append(
            Passage(
                label=p["label"],
                start=float(p["start"]),
                end=float(p["end"]),
                vocabulary=vocab,
                sequence=seq,
                source=p.get("source", ""),
                confidence=p.get("confidence", ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def cluster_notes(notes: list[dict], gap_s: float = _CLUSTER_GAP_S) -> list[set[tuple[int, int]]]:
    """Group notes whose onsets fall within ``gap_s`` into chords, in time order."""
    ordered = sorted(notes, key=lambda n: n["start"])
    clusters: list[set[tuple[int, int]]] = []
    last: float | None = None
    for n in ordered:
        pos = (int(n["string"]), int(n["fret"]))
        if last is not None and n["start"] - last <= gap_s:
            clusters[-1].add(pos)
        else:
            clusters.append({pos})
        last = n["start"]
    return clusters


def align_sequences(
    ref: list[set[tuple[int, int]]], hyp: list[set[tuple[int, int]]]
) -> tuple[int, int, int]:
    """Needleman-Wunsch over chord sets; returns (matched_notes, ref_notes, hyp_notes).

    Matching two chords scores the size of their intersection; a gap scores 0.
    The alignment that maximizes matched notes is used."""
    n, m = len(ref), len(hyp)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(
                score[i - 1][j - 1] + len(ref[i - 1] & hyp[j - 1]),
                score[i - 1][j],
                score[i][j - 1],
            )
    return score[n][m], sum(len(c) for c in ref), sum(len(c) for c in hyp)


def score_passage(
    passage: Passage, notes: list[dict], open_midis: tuple[int, ...]
) -> PassageResult:
    window = [n for n in notes if passage.start <= n["start"] < passage.end]
    count = len(window)
    vocab_pitches = {open_midis[s] + f for s, f in passage.vocabulary}
    max_fret = max((f for _, f in passage.vocabulary), default=0)
    if count:
        in_pos = sum(1 for n in window if (int(n["string"]), int(n["fret"])) in passage.vocabulary)
        in_pitch = sum(1 for n in window if int(n["pitch"]) in vocab_pitches)
        strays = sum(1 for n in window if int(n["fret"]) > max_fret + 2)
        seen = {(int(n["string"]), int(n["fret"])) for n in window}
        recall = (
            len(seen & passage.vocabulary) / len(passage.vocabulary) if passage.vocabulary else None
        )
        pos_p, pitch_p = in_pos / count, in_pitch / count
    else:
        pos_p = pitch_p = recall = None
        strays = 0
    seq_f1 = None
    details: dict = {}
    if passage.sequence is not None and count:
        matched, nref, nhyp = align_sequences(passage.sequence, cluster_notes(window))
        p = matched / nhyp if nhyp else 0.0
        r = matched / nref if nref else 0.0
        seq_f1 = 2 * p * r / (p + r) if p + r else 0.0
        details["sequence"] = {"matched": matched, "ref_notes": nref, "hyp_notes": nhyp}
    wrong = {}
    for n in window:
        pos = (int(n["string"]), int(n["fret"]))
        if pos not in passage.vocabulary:
            wrong[pos] = wrong.get(pos, 0) + 1
    details["most_common_wrong_positions"] = [
        {"string": s, "fret": f, "count": c}
        for (s, f), c in sorted(wrong.items(), key=lambda kv: -kv[1])[:6]
    ]
    return PassageResult(
        label=passage.label,
        note_count=count,
        position_precision=None if pos_p is None else round(pos_p, 3),
        pitch_precision=None if pitch_p is None else round(pitch_p, 3),
        vocab_recall=None if recall is None else round(recall, 3),
        high_fret_strays=strays,
        sequence_f1=None if seq_f1 is None else round(seq_f1, 3),
        details=details,
    )


def _open_midis(paths: VideoPaths) -> tuple[int, ...]:
    if paths.tuning_json.exists():
        d = json.loads(paths.tuning_json.read_text())
        midis = d.get("strings_midi")
        capo = int(d.get("capo", 0) or 0)
        if midis and len(midis) == 6:
            return tuple(int(m) + capo for m in midis)
    return _STANDARD_OPEN


def evaluate_video(paths: VideoPaths, eval_dir: Path = EVAL_DIR) -> dict:
    """Score frets.json against the video's reference passages."""
    passages = load_passages(paths.video_id, eval_dir)
    if not paths.frets_json.exists():
        raise FileNotFoundError(f"frets.json not found at {paths.frets_json}; run frets first")
    notes = json.loads(paths.frets_json.read_text())["notes"]
    open_midis = _open_midis(paths)
    results = [score_passage(p, notes, open_midis) for p in passages]
    scored = [r for r in results if r.note_count]
    total = sum(r.note_count for r in scored)

    def wmean(attr: str) -> float | None:
        vals = [(getattr(r, attr), r.note_count) for r in scored if getattr(r, attr) is not None]
        if not vals:
            return None
        return round(sum(v * n for v, n in vals) / sum(n for _, n in vals), 3)

    return {
        "video_id": paths.video_id,
        "passages": [
            {
                "label": r.label,
                "note_count": r.note_count,
                "position_precision": r.position_precision,
                "pitch_precision": r.pitch_precision,
                "vocab_recall": r.vocab_recall,
                "high_fret_strays": r.high_fret_strays,
                "sequence_f1": r.sequence_f1,
                **r.details,
            }
            for r in results
        ],
        "summary": {
            "passages": len(results),
            "notes": total,
            "position_precision": wmean("position_precision"),
            "pitch_precision": wmean("pitch_precision"),
            "vocab_recall": wmean("vocab_recall"),
            "high_fret_strays": sum(r.high_fret_strays for r in scored),
            "sequence_f1": wmean("sequence_f1"),
        },
    }
