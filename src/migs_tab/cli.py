"""migs-tab command line interface — pure plumbing, no LLM calls.

The LLM-driven steps (tips extraction, section detection, fret optimization,
vision disambiguation) are orchestrated by the Claude skill at
.claude/skills/migs-tab/SKILL.md, which runs against the user's Claude
subscription instead of the API.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console

from . import chord_shapes as chord_shapes_mod
from . import download as download_mod
from . import frames as frames_mod
from . import fret as fret_mod
from . import render as render_mod
from . import separate as separate_mod
from . import structure as structure_mod
from . import transcribe as transcribe_mod
from . import tuning as tuning_mod
from .paths import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, VideoPaths, extract_video_id

app = typer.Typer(
    help="Convert YouTube acoustic guitar tutorials into accurate tabs (plumbing layer).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _make_paths(url: str, cache_dir: Path) -> VideoPaths:
    return VideoPaths(extract_video_id(url), cache_dir=cache_dir)


@app.command()
def download(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Download video, audio (WAV), and captions."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Downloading[/bold] {paths.video_id}")
    download_mod.download(url, paths, force=force)
    _print_status(paths)


@app.command()
def clip(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    start: float = typer.Option(..., "--start", help="Start time in seconds"),
    duration: float = typer.Option(..., "--duration", help="Duration in seconds"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    name: str = typer.Option("clip", "--name", help="Name for the clipped audio file"),
) -> None:
    """Slice a segment out of the downloaded audio. Useful for fast iteration."""
    paths = _make_paths(url, cache_dir)
    if not paths.audio.exists():
        raise typer.BadParameter(f"audio.wav not found for {paths.video_id} — run download first")

    out = paths.root / f"{name}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(paths.audio),
        "-ar",
        "44100",
        "-ac",
        "2",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    console.print(f"[green]✓[/green] wrote clip → {out}")


@app.command()
def separate(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    model: str = typer.Option("htdemucs", "--model"),
    force: bool = typer.Option(False, "--force"),
    audio_name: str = typer.Option(
        "audio", "--audio-name", help="Which cached wav to separate (default: 'audio')"
    ),
) -> None:
    """Isolate the guitar stem with Demucs."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Separating[/bold] {paths.video_id} ({audio_name}.wav) with {model}")
    separate_mod.separate(paths, model=model, force=force, audio_name=audio_name)
    _print_status(paths)


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Transcribe the isolated guitar stem to MIDI + JSON via basic-pitch."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Transcribing[/bold] {paths.video_id}")
    transcribe_mod.transcribe(paths, force=force)
    _print_status(paths)


@app.command()
def structure(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Analyze playing segments + chord progressions; write structure.json."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Analyzing structure[/bold] for {paths.video_id}")
    structure_mod.analyze_structure(paths, force=force)
    _print_status(paths)


@app.command()
def tuning(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Detect tuning + capo (captions first, audio second); write tuning.json."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Detecting tuning[/bold] for {paths.video_id}")
    tuning_mod.detect_tuning(paths, force=force)
    import json as _json

    data = _json.loads(paths.tuning_json.read_text())
    console.print(
        f"  [green]✓[/green] {data['label']}  (capo {data['capo']}, "
        f"source={data['source']}, confidence={data['confidence']})"
    )
    console.print(f"  evidence: {data['evidence']}")


@app.command()
def frets(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Assign string/fret positions to every note (Viterbi); write frets.json."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Assigning frets[/bold] for {paths.video_id}")
    fret_mod.assign_frets(paths, force=force)
    _print_status(paths)


@app.command()
def frame(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    timestamp: float = typer.Argument(..., help="Time in seconds to grab a frame"),
    label: str = typer.Option("", "--label", help="Suffix to append to the filename"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
) -> None:
    """Extract a single still frame from the video at the given timestamp."""
    paths = _make_paths(url, cache_dir)
    out = frames_mod.extract_frame(paths, timestamp, label=label or None)
    console.print(f"[green]✓[/green] {out}")


@app.command(name="frames-for-clusters")
def frames_for_clusters(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cluster_ids: str = typer.Argument(..., help="Comma-separated cluster IDs from frets.json"),
    max_frames: int = typer.Option(
        10,
        "--max-frames",
        help="Hard ceiling on the number of frames extracted. Protects against runaway vision-pass costs.",
    ),
    subdir: str = typer.Option("ambiguous", "--subdir"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
) -> None:
    """Extract one video frame per cluster ID for the LLM vision pass."""
    paths = _make_paths(url, cache_dir)
    ids = [int(s) for s in cluster_ids.split(",") if s.strip()]
    if not ids:
        raise typer.BadParameter("provide at least one cluster id")
    result = frames_mod.extract_frames_for_clusters(
        paths, ids, max_frames=max_frames, subdir=subdir
    )
    console.print(
        f"[green]✓[/green] {result['extracted_count']} frames written to "
        f"{paths.frames_dir / subdir}"
        + (
            f" ([yellow]{result['skipped_due_to_cap']} skipped[/yellow] due to --max-frames cap)"
            if result["skipped_due_to_cap"]
            else ""
        )
    )
    for rec in result["cluster_records"]:
        console.print(
            f"  cluster {rec['cluster_id']:>4}  onset {rec.get('onset', '?'):>7.2f}s  "
            f"→ {rec.get('frame_path', '<error>')}"
        )


@app.command(name="chord-shape-frames")
def chord_shape_frames(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Extract one representative frame per unique chord (for vision verification)."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Extracting chord-shape frames[/bold] for {paths.video_id}")
    out_path = chord_shapes_mod.select_and_extract(paths, force=force)
    import json as _json

    data = _json.loads(out_path.read_text())
    console.print(f"  [green]✓[/green] {data['chord_count']} unique chord(s) → {out_path}")
    for chord, info in data["candidates"].items():
        console.print(
            f"     {chord:<8} @ {info['primary_timestamp']:>7.2f}s  "
            f"(sharpness {info.get('primary_sharpness', 0):.0f}, "
            f"{info.get('primary_span_duration', 0):.1f}s span, "
            f"{info['total_instances']} total instance(s))"
        )


@app.command()
def render(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
    line_width: int = typer.Option(72, "--line-width"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Render the section-by-section ASCII tab to output/<id>/tab.txt + tab.md."""
    paths = _make_paths(url, cache_dir)
    out = render_mod.render(paths, output_root=output_dir, line_width=line_width, force=force)
    console.print(f"[green]✓[/green] {out}")


@app.command()
def process(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run the plumbing pipeline: download → separate → transcribe → structure → frets."""
    paths = _make_paths(url, cache_dir)
    console.rule(f"[bold cyan]migs-tab • {paths.video_id}")

    console.print("[bold]1/6[/bold] download")
    download_mod.download(url, paths, force=force)

    console.print("[bold]2/6[/bold] separate (Demucs)")
    separate_mod.separate(paths, force=force)

    console.print("[bold]3/6[/bold] transcribe (basic-pitch)")
    transcribe_mod.transcribe(paths, force=force)

    console.print("[bold]4/6[/bold] structure (librosa)")
    structure_mod.analyze_structure(paths, force=force)

    console.print("[bold]5/6[/bold] tuning")
    tuning_mod.detect_tuning(paths, force=force)

    console.print("[bold]6/6[/bold] frets (Viterbi)")
    fret_mod.assign_frets(paths, force=force)

    console.rule("[bold green]done — LLM steps now run via the /migs-tab skill")
    _print_status(paths)


@app.command()
def status(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
) -> None:
    """Show which artifacts are cached for a given video."""
    _print_status(_make_paths(url, cache_dir))


def _print_status(paths: VideoPaths) -> None:
    items: list[tuple[str, Path]] = [
        ("video", paths.video),
        ("audio (wav)", paths.audio),
        ("captions (vtt)", paths.captions_vtt),
        ("captions (txt)", paths.captions_text),
        ("guitar stem", paths.guitar_stem),
        ("notes.mid", paths.notes_midi),
        ("notes.json", paths.notes_json),
        ("structure.json", paths.structure_json),
        ("sections.json", paths.sections_json),
        ("tuning.json", paths.tuning_json),
        ("frets.json", paths.frets_json),
        ("tips.md", paths.tips_md),
    ]
    console.print(f"\n[bold]Cache:[/bold] {paths.root}")
    for label, path in items:
        mark = "[green]✓[/green]" if path.exists() else "[dim]·[/dim]"
        console.print(f"  {mark} {label:<18} {path.name}")


if __name__ == "__main__":
    app()
