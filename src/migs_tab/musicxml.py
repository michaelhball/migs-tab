"""MusicXML export of the rendered tab.

Writes a MusicXML 3.1 file that opens in MuseScore, TuxGuitar, Guitar Pro,
and most other notation software. The exporter consumes the same inputs as
the ASCII tab renderer (sections + frets + beat grid) so both stay in sync.

The XML structure is intentionally minimal but valid:
 - one ``<part>`` per song (all sections concatenated, with section headers
   embedded as rehearsal marks),
 - 4/4 time, divisions=2 (so each 8th-note subdivision is one ``<duration>``
   unit),
 - tab clef with per-string tuning derived from the detected tuning,
 - notes carry both ``<pitch>`` AND ``<technical><string>``+``<fret>`` so
   notation editors can show either staff-notation or tab views.

Duration quantization: every onset cluster fires on a single 8th-note
subdivision (the closest one to its actual onset). Cluster notes are
emitted as a chord (one note per pitch, all but the first carry the
``<chord/>`` flag). Subdivisions with no cluster are filled with rests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, tostring

if TYPE_CHECKING:
    from .render import RenderedSection, TuningInfo

# Pitch-class step letters as MusicXML wants them, with the alter (sharp).
_STEP_FROM_PC: list[tuple[str, int]] = [
    ("C", 0),
    ("C", 1),
    ("D", 0),
    ("D", 1),
    ("E", 0),
    ("F", 0),
    ("F", 1),
    ("G", 0),
    ("G", 1),
    ("A", 0),
    ("A", 1),
    ("B", 0),
]


@dataclass
class _NoteEvent:
    """One note at a quantized position. ``string`` is the MusicXML convention
    (1 = highest-pitched string), NOT our internal 0-low-E convention.

    ``marks`` are articulation notations to attach: tuples of
    (kind, arg) where kind ∈ {hammer, pull, slide, bend, harmonic} and
    arg is "start"/"stop" for the pairwise kinds, the bend's
    target_semitones, or None for harmonics."""

    midi: int
    mx_string: int
    fret: int
    marks: tuple = ()


def _midi_to_step_octave_alter(midi: int) -> tuple[str, int, int]:
    """Return (step, octave, alter) for a MIDI pitch.

    MIDI 60 = middle C = C4. Step is the letter name; alter is +1 for sharps,
    -1 for flats, 0 otherwise. We always emit sharps for simplicity.
    """
    pc = midi % 12
    step, alter = _STEP_FROM_PC[pc]
    # MusicXML octave is concert pitch octave (C4 = middle C, MIDI 60).
    octave = (midi // 12) - 1
    return step, octave, alter


def render_musicxml(
    sections_data: dict,
    rendered: list[RenderedSection],
    tuning: TuningInfo,
    notes_by_section: dict[str, list[dict]],
    beat_times_by_section: dict[str, list[float]],
    subdivisions_per_beat: int = 2,
    beats_per_bar: int = 4,
    articulations: list[dict] | None = None,
) -> bytes:
    """Build a MusicXML score-partwise XML byte string.

    Caller is responsible for filtering ``notes_by_section`` to the
    canonical instance's time window per section. ``articulations``
    (frets.json's optional top-level list; entries reference notes by
    global note_index) adds slur + hammer-on/pull-off, slide, bend and
    natural-harmonic notations; None/empty leaves the output
    byte-identical to before the articulations feature.
    """
    marks_by_idx = _articulation_marks(articulations, notes_by_section)
    score = Element("score-partwise", version="3.1")

    work = SubElement(score, "work")
    SubElement(work, "work-title").text = f"migs-tab — {sections_data.get('video_id', '?')}"

    ident = SubElement(score, "identification")
    encoding = SubElement(ident, "encoding")
    SubElement(encoding, "software").text = "migs-tab"

    part_list = SubElement(score, "part-list")
    score_part = SubElement(part_list, "score-part", id="P1")
    SubElement(score_part, "part-name").text = f"Guitar ({tuning.label})"

    part = SubElement(score, "part", id="P1")

    measure_num = 1
    first_measure = True
    for sec in rendered:
        notes = notes_by_section.get(sec.label, [])
        beats = beat_times_by_section.get(sec.label, [])
        if not beats:
            continue

        # Expand beats to subdivisions (each subdivision = one 8th note).
        subdivisions = _subdivisions(beats, subdivisions_per_beat)
        if len(subdivisions) < 2:
            continue

        # Map each note to its nearest subdivision index.
        note_events_by_slot: dict[int, list[_NoteEvent]] = {}
        for n in notes:
            slot = _nearest_index([t for t, _ in subdivisions], n["start"])
            # Convert our internal string index (0 = low E) to MusicXML
            # convention (1 = highest-pitched string).
            mx_string = 6 - int(n["string"])
            note_events_by_slot.setdefault(slot, []).append(
                _NoteEvent(
                    midi=int(n["pitch"]),
                    mx_string=mx_string,
                    fret=int(n["fret"]),
                    marks=tuple(marks_by_idx.get(n.get("note_index"), ())),
                )
            )

        # Emit measures of beats_per_bar * subdivisions_per_beat slots each.
        slots_per_measure = beats_per_bar * subdivisions_per_beat
        # Each subdivision = one 8th note = 1 duration unit when divisions=2.
        duration_per_slot = 1
        bpm = sec.tempo_bpm if sec.tempo_bpm > 0 else 90.0

        for slot_start in range(0, len(subdivisions), slots_per_measure):
            measure = SubElement(part, "measure", number=str(measure_num))
            if first_measure:
                _emit_attributes(measure, tuning, divisions=2, beats=beats_per_bar)
                _emit_tempo_direction(measure, bpm)
                first_measure = False

            # Section header on the first measure of each section.
            if slot_start == 0:
                _emit_rehearsal(measure, sec.label)

            for offset in range(slots_per_measure):
                slot_idx = slot_start + offset
                if slot_idx >= len(subdivisions):
                    # Pad the final partial measure with rests.
                    _emit_rest(measure, duration_per_slot)
                    continue
                events = note_events_by_slot.get(slot_idx, [])
                if not events:
                    _emit_rest(measure, duration_per_slot)
                else:
                    for i, ev in enumerate(events):
                        _emit_note(
                            measure,
                            ev,
                            duration=duration_per_slot,
                            is_chord=(i > 0),
                        )

            measure_num += 1

    return b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + tostring(
        score, encoding="utf-8"
    )


def _articulation_marks(
    articulations: list[dict] | None,
    notes_by_section: dict[str, list[dict]],
) -> dict[int, list[tuple[str, object]]]:
    """Map global note_index → articulation marks to attach to its <note>.

    Hammer/pull pairs become a <slur> start/stop plus
    <technical><hammer-on>/<pull-off> start/stop; slides a <slide>
    start/stop; bends a <technical><bend><bend-alter>; harmonics a
    <technical><harmonic><natural/>. A pair is marked only when BOTH
    halves are being emitted from the SAME section — a dangling slur
    start (other half filtered out, or in a differently-ordered section)
    is invalid notation.

    Each mark is additionally gated on the EMITTED note records matching
    the articulation's own (string, fret) evidence — the same
    on-expected-string / measured-position checks the ASCII renderer
    applies. A verified-chord-shape override that re-strings a pair half
    voids the evidence (a cross-string hammer-on is impossible notation)
    and a bend's struck note moved off the measured position voids the
    bend; both are silently dropped here, keeping tab.musicxml in
    agreement with tab.txt from the same run. Malformed entries are
    skipped, never raised.
    """
    if not articulations:
        return {}
    # note_index → (section label, emitted note record). Last-write-wins
    # if the same note_index ever appeared in two sections — safe today
    # because canonical instance windows never overlap (each note's start
    # time selects at most one section); if overlapping windows ever
    # become legal this must become a per-section map, since marks are
    # attached globally by note_index and a pair valid in one section
    # would dangle in the other.
    emitted: dict[int, tuple[str, dict]] = {}
    for label, notes in notes_by_section.items():
        for n in notes:
            idx = n.get("note_index")
            if idx is not None:
                emitted[int(idx)] = (label, n)
    marks: dict[int, list[tuple[str, object]]] = {}
    for a in articulations:
        typ = a.get("type")
        try:
            if typ in ("hammer", "pull", "slide"):
                from_idx = int(a["from_note_index"])
                to_idx = int(a["note_index"])
                a_string = int(a["string"])
                src = emitted.get(from_idx)
                dst = emitted.get(to_idx)
                if (
                    src is not None
                    and dst is not None
                    and src[0] == dst[0]
                    and int(src[1]["string"]) == a_string
                    and int(dst[1]["string"]) == a_string
                    and int(src[1]["fret"]) == int(a["from_fret"])
                    and int(dst[1]["fret"]) == int(a["to_fret"])
                ):
                    marks.setdefault(from_idx, []).append((typ, "start"))
                    marks.setdefault(to_idx, []).append((typ, "stop"))
            elif typ == "bend":
                idx = int(a["note_index"])
                rec = emitted.get(idx)
                if (
                    rec is not None
                    and int(rec[1]["string"]) == int(a["string"])
                    and int(rec[1]["fret"]) == int(a["fret"])
                ):
                    marks.setdefault(idx, []).append((typ, int(a.get("target_semitones", 1))))
            elif typ == "harmonic":
                idx = int(a["note_index"])
                rec = emitted.get(idx)
                # The prelayout pass re-strings a valid harmonic to
                # (open_string, node_fret); a note NOT at that position
                # means the entry was rejected there (or never applied)
                # and must not be marked here either.
                if (
                    rec is not None
                    and int(rec[1]["string"]) == int(a["open_string"])
                    and int(rec[1]["fret"]) == int(a["node_fret"])
                ):
                    marks.setdefault(idx, []).append((typ, None))
        except (KeyError, TypeError, ValueError):
            continue
    return marks


def _subdivisions(beat_times: list[float], subs_per_beat: int) -> list[tuple[float, bool]]:
    """Expand beats into subdivisions. Returns list of (time, is_beat)."""
    if len(beat_times) < 2:
        return [(t, True) for t in beat_times]
    grid: list[tuple[float, bool]] = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        step = (beat_end - beat_start) / subs_per_beat
        for j in range(subs_per_beat):
            grid.append((beat_start + j * step, j == 0))
    grid.append((beat_times[-1], True))
    return grid


def _nearest_index(sorted_times: list[float], t: float) -> int:
    import bisect

    i = bisect.bisect_left(sorted_times, t)
    if i == 0:
        return 0
    if i >= len(sorted_times):
        return len(sorted_times) - 1
    before = sorted_times[i - 1]
    after = sorted_times[i]
    return i - 1 if abs(t - before) <= abs(t - after) else i


def _emit_attributes(measure: Element, tuning: TuningInfo, divisions: int, beats: int) -> None:
    attrs = SubElement(measure, "attributes")
    SubElement(attrs, "divisions").text = str(divisions)
    key = SubElement(attrs, "key")
    SubElement(key, "fifths").text = "0"
    time = SubElement(attrs, "time")
    SubElement(time, "beats").text = str(beats)
    SubElement(time, "beat-type").text = "4"
    clef = SubElement(attrs, "clef")
    SubElement(clef, "sign").text = "TAB"
    SubElement(clef, "line").text = "5"
    # Per-string tuning. staff-tuning line=1 is the BOTTOM line of the tab
    # (i.e. lowest-pitched string). tuning.strings_midi is low→high so the
    # mapping is line(i+1) = strings_midi[i].
    sd = SubElement(attrs, "staff-details")
    SubElement(sd, "staff-lines").text = "6"
    for i, midi in enumerate(tuning.strings_midi):
        step, octave, _ = _midi_to_step_octave_alter(midi)
        st = SubElement(sd, "staff-tuning", line=str(i + 1))
        SubElement(st, "tuning-step").text = step
        SubElement(st, "tuning-octave").text = str(octave)
    if tuning.capo:
        SubElement(sd, "capo").text = str(tuning.capo)


def _emit_tempo_direction(measure: Element, bpm: float) -> None:
    direction = SubElement(measure, "direction", placement="above")
    dtype = SubElement(direction, "direction-type")
    metronome = SubElement(dtype, "metronome")
    SubElement(metronome, "beat-unit").text = "quarter"
    SubElement(metronome, "per-minute").text = f"{bpm:.0f}"
    sound = SubElement(direction, "sound", tempo=f"{bpm:.0f}")  # noqa: F841


def _emit_rehearsal(measure: Element, label: str) -> None:
    direction = SubElement(measure, "direction", placement="above")
    dtype = SubElement(direction, "direction-type")
    SubElement(dtype, "rehearsal").text = label


def _emit_rest(measure: Element, duration: int) -> None:
    note = SubElement(measure, "note")
    SubElement(note, "rest")
    SubElement(note, "duration").text = str(duration)
    SubElement(note, "voice").text = "1"
    SubElement(note, "type").text = "eighth"


def _emit_note(measure: Element, ev: _NoteEvent, duration: int, is_chord: bool) -> None:
    note_el = SubElement(measure, "note")
    if is_chord:
        SubElement(note_el, "chord")
    pitch = SubElement(note_el, "pitch")
    step, octave, alter = _midi_to_step_octave_alter(ev.midi)
    SubElement(pitch, "step").text = step
    if alter != 0:
        SubElement(pitch, "alter").text = str(alter)
    SubElement(pitch, "octave").text = str(octave)
    SubElement(note_el, "duration").text = str(duration)
    SubElement(note_el, "voice").text = "1"
    SubElement(note_el, "type").text = "eighth"
    notations = SubElement(note_el, "notations")
    technical = SubElement(notations, "technical")
    SubElement(technical, "string").text = str(ev.mx_string)
    SubElement(technical, "fret").text = str(ev.fret)
    # Articulation notations. Children of <notations> and <technical> are
    # repeated choice groups in the MusicXML schema, so appending after
    # string/fret is valid.
    for kind, arg in ev.marks:
        if kind in ("hammer", "pull"):
            SubElement(notations, "slur", number="1", type=str(arg))
            tech_el = SubElement(
                technical, "hammer-on" if kind == "hammer" else "pull-off", type=str(arg)
            )
            if arg == "start":
                tech_el.text = "H" if kind == "hammer" else "P"
        elif kind == "slide":
            SubElement(
                notations,
                "slide",
                attrib={"line-type": "solid", "number": "1", "type": str(arg)},
            )
        elif kind == "bend":
            bend_el = SubElement(technical, "bend")
            SubElement(bend_el, "bend-alter").text = str(arg)
        elif kind == "harmonic":
            harm_el = SubElement(technical, "harmonic")
            SubElement(harm_el, "natural")
