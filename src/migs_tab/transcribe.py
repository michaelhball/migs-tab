"""Polyphonic note transcription via basic-pitch.

Operates on the isolated guitar stem (preferred) or the raw audio as a fallback.
Produces a MIDI file plus a JSON note list keyed by onset/offset/pitch/velocity,
which downstream phases (section detection, fret optimization) consume.
"""

from __future__ import annotations

import json
from pathlib import Path

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict_and_save

from .paths import VideoPaths

# Acoustic-guitar-friendly defaults. Standard tuning ranges E2 (40) to ~D6 (86).
# We give some headroom for capoed / drop tunings and high-register lead lines.
_GUITAR_MIDI_MIN = 38  # D2
_GUITAR_MIDI_MAX = 88  # E6

# basic-pitch thresholds — these are tuned for general music; we tighten the
# onset threshold a hair to avoid false positives from string noise / breath.
_ONSET_THRESHOLD = 0.5
_FRAME_THRESHOLD = 0.3
_MIN_NOTE_LENGTH_MS = 58  # one frame at basic-pitch's hop size


def transcribe(paths: VideoPaths, force: bool = False) -> VideoPaths:
    """Run basic-pitch on the isolated guitar stem, write notes.mid + notes.json."""
    if paths.notes_midi.exists() and paths.notes_json.exists() and not force:
        return paths

    source = paths.guitar_stem if paths.guitar_stem.exists() else paths.audio
    if not source.exists():
        raise FileNotFoundError(
            f"No audio available for transcription ({paths.guitar_stem} / {paths.audio})"
        )

    # predict_and_save handles file I/O for MIDI + (optional) sonification + CSV.
    # We point it at the cache dir, then re-emit notes.json from the MIDI for our own use.
    out_dir = paths.root
    predict_and_save(
        audio_path_list=[str(source)],
        output_directory=str(out_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=Model(ICASSP_2022_MODEL_PATH),
        onset_threshold=_ONSET_THRESHOLD,
        frame_threshold=_FRAME_THRESHOLD,
        minimum_note_length=_MIN_NOTE_LENGTH_MS,
        minimum_frequency=_midi_to_hz(_GUITAR_MIDI_MIN),
        maximum_frequency=_midi_to_hz(_GUITAR_MIDI_MAX),
        multiple_pitch_bends=False,
        melodia_trick=True,
    )

    # basic-pitch names the file <stem>_basic_pitch.mid. Normalize to notes.mid.
    bp_midi = out_dir / f"{source.stem}_basic_pitch.mid"
    if bp_midi.exists():
        bp_midi.replace(paths.notes_midi)

    _write_notes_json(paths.notes_midi, paths.notes_json)

    return paths


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _write_notes_json(midi_path: Path, json_path: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[dict] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            notes.append(
                {
                    "start": round(float(note.start), 4),
                    "end": round(float(note.end), 4),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                }
            )
    notes.sort(key=lambda n: (n["start"], n["pitch"]))

    json_path.write_text(
        json.dumps(
            {
                "source_midi": midi_path.name,
                "note_count": len(notes),
                "notes": notes,
            },
            indent=2,
        )
    )
