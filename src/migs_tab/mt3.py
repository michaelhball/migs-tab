"""YourMT3+ transcription backend.

Wraps the vendored YourMT3+ checkpointed model (at ``third_party/YourMT3/``)
as an alternative to basic-pitch. Runs in a dedicated venv via subprocess so
its pinned deps (``numpy==1.26.4``, ``transformers==4.45.1``, ``lightning``)
don't infect the main migs-tab env.

The driver script (``third_party/YourMT3/migs_driver.py``) selects MPS when
available, falls back to CUDA, then CPU. We post-process its MIDI output the
same way the basic-pitch path does so downstream phases stay backend-agnostic.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .paths import PROJECT_ROOT, VideoPaths

_THIRD_PARTY_DIR = PROJECT_ROOT / "third_party" / "YourMT3"
_DRIVER_PATH = _THIRD_PARTY_DIR / "migs_driver.py"
_DEFAULT_VENV = _THIRD_PARTY_DIR / ".venv"
_DEFAULT_VARIANT = "YMT3+"
_DEFAULT_BATCH_SIZE = 2


class MT3NotInstalled(RuntimeError):
    """Raised when YourMT3+ isn't set up locally (no venv / no checkpoint)."""


def transcribe(
    paths: VideoPaths,
    *,
    force: bool = False,
    variant: str = _DEFAULT_VARIANT,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    venv: Path | None = None,
    audio_source: Path | None = None,
) -> VideoPaths:
    """Run YourMT3+ on a guitar stem (or audio_source override); write notes.mt3.mid + .json."""
    if paths.notes_mt3_midi.exists() and paths.notes_mt3_json.exists() and not force:
        return paths

    source = _resolve_audio_source(paths, audio_source)
    proc = transcribe_async(
        paths, variant=variant, batch_size=batch_size, venv=venv, audio_source=source
    )
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)
    _write_notes_json(
        paths.notes_mt3_midi,
        paths.notes_mt3_json,
        audio_source=source,
        cache_root=paths.root,
        variant=variant,
        batch_size=batch_size,
    )
    return paths


def transcribe_async(
    paths: VideoPaths,
    *,
    variant: str = _DEFAULT_VARIANT,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    venv: Path | None = None,
    audio_source: Path | None = None,
) -> subprocess.Popen:
    """Start MT3 in the background; caller is responsible for ``wait()`` and the JSON post-step.

    Use ``finalize_async()`` after the Popen's ``wait()`` returns 0 to write the notes JSON.
    """
    if not _DRIVER_PATH.exists():
        raise MT3NotInstalled(
            f"YourMT3+ driver missing at {_DRIVER_PATH}; clone the upstream Space first"
        )

    python = _resolve_python(venv)
    source = _resolve_audio_source(paths, audio_source)

    cmd = [
        str(python),
        str(_DRIVER_PATH),
        "--audio",
        str(source),
        "--output",
        str(paths.notes_mt3_midi),
        "--variant",
        variant,
        "--batch-size",
        str(batch_size),
    ]
    return subprocess.Popen(cmd, cwd=str(_THIRD_PARTY_DIR))


def finalize_async(
    paths: VideoPaths,
    *,
    audio_source: Path | None = None,
    variant: str | None = None,
    batch_size: int | None = None,
) -> None:
    """Write notes.mt3.json from the MIDI produced by a successful async run.

    Pass the same ``audio_source``/``variant``/``batch_size`` that were given
    to ``transcribe_async`` so the JSON records true provenance.
    """
    _write_notes_json(
        paths.notes_mt3_midi,
        paths.notes_mt3_json,
        audio_source=audio_source,
        cache_root=paths.root,
        variant=variant,
        batch_size=batch_size,
    )


def _resolve_audio_source(paths: VideoPaths, audio_source: Path | None) -> Path:
    """Pick the audio MT3 should transcribe: explicit override, else stem, else mix.

    Preferring the Demucs guitar stem matters: running MT3 on the raw mix
    transcribes the instructor's SPEECH as a "Singing Voice" note channel
    (670 notes on the SRV tutorial) and muddies every other channel.
    """
    if audio_source is not None:
        source = audio_source
    else:
        source = paths.guitar_stem if paths.guitar_stem.exists() else paths.audio
    if not source.exists():
        raise FileNotFoundError(f"No audio for MT3 transcription ({source})")
    return source


def _resolve_python(venv: Path | None) -> Path:
    """Pick the python interpreter to run the driver with."""
    candidate = venv or _DEFAULT_VENV
    py = candidate / "bin" / "python"
    if py.exists():
        return py
    system_py = shutil.which("python3")
    if system_py:
        return Path(system_py)
    raise MT3NotInstalled(
        f"No YourMT3+ venv found at {candidate} and no system python3 on PATH; "
        f"see third_party/YourMT3/README for setup"
    )


# General MIDI guitar programs: 24=Nylon, 25=Steel, 26=Jazz, 27=Clean,
# 28=Muted, 29=Overdrive, 30=Distortion, 31=Harmonics. Kept as a constant so
# tests / downstream code can refer to the canonical set, even though the
# filter below now accepts all non-drum melodic programs.
_GUITAR_PROGRAMS = frozenset(range(24, 32))

# YourMT3+'s MIDI writer (third_party/YourMT3/amt/src/utils/midi.py,
# note_event2midi) maps its internal "Singing Voice" program (100) to GM
# program 65 and "Singing Voice (chorus)" (101) to GM program 53, and always
# names the track from its vocabulary. On a guitar tutorial the voice channel
# is the instructor TALKING transcribed as notes (670 of them on the SRV
# video) — never guitar — so it is the only non-drum channel that is safe to
# drop wholesale. Every other non-guitar-labeled melodic channel (Piano,
# Strings, ...) is mislabeled real guitar audio and must be kept.
#
# The track NAME is the reliable discriminator: on model variants with
# full_plus vocabs, GM programs 53 ('Voice Oohs') and 65 ('Alto Sax') are
# legitimate — often guitar-mislabeled — outputs, so matching on program alone
# would wholesale-delete real guitar channels. The program set is consulted
# only when the track name is missing.
_VOICE_TRACK_PREFIX = "Singing Voice"
_VOICE_MIDI_PROGRAMS = frozenset({53, 65})

# Plausible guitar pitch range, inclusive MIDI note numbers. Low bound 36 (C2)
# covers Drop D a half-step down (C#2 = 37) with a semitone of margin; high
# bound 88 (E6) is the 24th fret of the high E string in standard tuning.
# Cached MT3 output spans MIDI 26-100 — anything outside this window is a
# transcription artifact, not a playable guitar note.
_MIN_GUITAR_MIDI = 36
_MAX_GUITAR_MIDI = 88

# Dedup tolerance: two notes with the same pitch firing within this window are
# treated as the same physical attack. Matches the basic-pitch onset hop and
# the A/B comparator we used to characterize the model.
_DEDUP_ONSET_TOL_S = 0.05


def _is_voice_channel(name: str, program: int) -> bool:
    """True if an MT3 MIDI track is the singing-voice channel (instructor speech).

    The track name is authoritative — the driver always writes it from the
    model's inverse vocab. The GM program fallback only applies when the name
    is missing, because programs 53/65 are legitimate (often guitar-mislabeled)
    outputs on full_plus-vocab variants.
    """
    if name:
        return name.startswith(_VOICE_TRACK_PREFIX)
    return program in _VOICE_MIDI_PROGRAMS


def _provenance(
    audio_source: Path | None,
    cache_root: Path | None,
    variant: str | None,
    batch_size: int | None,
) -> dict:
    """Record which audio MT3 transcribed plus model parameters.

    Angie's cached run used the clean stem while Life-by-the-Drop's used the
    raw mix and nothing recorded which — that ambiguity derailed a whole
    investigation. Path is stored relative to the cache dir when possible.
    """
    source_str: str | None = None
    mtime_iso: str | None = None
    size_bytes: int | None = None
    if audio_source is not None:
        source_str = str(audio_source)
        if cache_root is not None:
            try:
                source_str = str(audio_source.relative_to(cache_root))
            except ValueError:
                pass
        if audio_source.exists():
            st = audio_source.stat()
            mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(timespec="seconds")
            size_bytes = int(st.st_size)
    return {
        "audio_source": source_str,
        "audio_mtime": mtime_iso,
        "audio_size_bytes": size_bytes,
        "model_variant": variant,
        "batch_size": batch_size,
    }


def _write_notes_json(
    midi_path: Path,
    json_path: Path,
    *,
    audio_source: Path | None = None,
    cache_root: Path | None = None,
    variant: str | None = None,
    batch_size: int | None = None,
) -> None:
    """Parse MT3-emitted MIDI into the canonical {start,end,pitch,velocity} list.

    YourMT3+ hallucinates instrument channels even when the input is a single
    guitar stem — it routes strum attacks to "Drums" (GM drum codes 37/38/42),
    doubles up real notes between "Guitar" and "Acoustic/Electric Piano", and
    occasionally fires "Chromatic Percussion" / "Bass" on guitar notes in the
    expected range.

    Channel audit on a 30s Angie slice showed:
      - 72% of non-guitar notes were exact duplicates of the Guitar channel
      - 83% were confirmed by basic-pitch
      - Drum-channel pitches are GM kit codes (side-stick/snare/hi-hat), never
        useful musical pitches

    So we keep every non-drum channel except "Singing Voice" (instructor
    speech — see _VOICE_TRACK_PREFIX), drop notes outside the playable guitar
    range (_MIN_GUITAR_MIDI.._MAX_GUITAR_MIDI), then dedupe by (pitch,
    start ±50ms) to collapse the cross-channel duplicates. Keeping all other
    melodic channels recovers ~6% of real guitar notes that the previous
    "guitar-programs-only" filter dropped. Everything filtered is counted in
    the output JSON so nothing disappears silently.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    breakdown: list[dict] = []
    collected: list[dict] = []
    voice_channels: list[dict] = []
    voice_note_count = 0
    out_of_range_count = 0
    for instrument in pm.instruments:
        breakdown.append(
            {
                "program": int(instrument.program),
                "name": instrument.name or "?",
                "is_drum": bool(instrument.is_drum),
                "note_count": len(instrument.notes),
            }
        )
        if instrument.is_drum:
            continue
        if _is_voice_channel(instrument.name or "", int(instrument.program)):
            voice_note_count += len(instrument.notes)
            voice_channels.append(
                {
                    "program": int(instrument.program),
                    "name": instrument.name or "?",
                    "note_count": len(instrument.notes),
                }
            )
            continue
        for note in instrument.notes:
            if not _MIN_GUITAR_MIDI <= int(note.pitch) <= _MAX_GUITAR_MIDI:
                out_of_range_count += 1
                continue
            collected.append(
                {
                    "start": round(float(note.start), 4),
                    "end": round(float(note.end), 4),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                }
            )
    collected.sort(key=lambda n: (n["start"], n["pitch"]))
    notes = _dedup_notes(collected, _DEDUP_ONSET_TOL_S)

    json_path.write_text(
        json.dumps(
            {
                "source_midi": midi_path.name,
                "backend": "mt3",
                "note_count": len(notes),
                "filter": (
                    "voice-drop + pitch-clamp "
                    f"{_MIN_GUITAR_MIDI}-{_MAX_GUITAR_MIDI} + "
                    "all-non-drum-melodic + (pitch, start±50ms) dedupe"
                ),
                "pitch_range": [_MIN_GUITAR_MIDI, _MAX_GUITAR_MIDI],
                "filtered_voice_notes": voice_note_count,
                "filtered_voice_channels": voice_channels,
                "filtered_out_of_range_notes": out_of_range_count,
                "provenance": _provenance(audio_source, cache_root, variant, batch_size),
                "instrument_breakdown": breakdown,
                "notes": notes,
            },
            indent=2,
        )
    )


def _dedup_notes(notes: list[dict], tol_s: float) -> list[dict]:
    """Drop notes that share pitch with an earlier-kept note within ``tol_s`` of its start.

    Earlier-start wins; the kept note's velocity is raised to the loudest of
    its merged duplicates. We keep the onset of the first instance and rely on
    its end time — channel duplicates have effectively identical timing so the
    end discrepancy is sub-frame.
    """
    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        bucket = by_pitch.setdefault(n["pitch"], [])
        merged = False
        for kept in bucket:
            if abs(n["start"] - kept["start"]) <= tol_s:
                if n["velocity"] > kept["velocity"]:
                    kept["velocity"] = n["velocity"]
                merged = True
                break
        if not merged:
            bucket.append(n)
    out: list[dict] = [n for bucket in by_pitch.values() for n in bucket]
    out.sort(key=lambda n: (n["start"], n["pitch"]))
    return out
