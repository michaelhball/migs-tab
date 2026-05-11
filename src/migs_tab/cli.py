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

from . import download as download_mod
from . import separate as separate_mod
from . import transcribe as transcribe_mod
from .paths import DEFAULT_CACHE_DIR, VideoPaths, extract_video_id

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
def process(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run the plumbing pipeline: download → separate → transcribe."""
    paths = _make_paths(url, cache_dir)
    console.rule(f"[bold cyan]migs-tab • {paths.video_id}")

    console.print("[bold]1/3[/bold] download")
    download_mod.download(url, paths, force=force)

    console.print("[bold]2/3[/bold] separate (Demucs)")
    separate_mod.separate(paths, force=force)

    console.print("[bold]3/3[/bold] transcribe (basic-pitch)")
    transcribe_mod.transcribe(paths, force=force)

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
        ("tips.md", paths.tips_md),
    ]
    console.print(f"\n[bold]Cache:[/bold] {paths.root}")
    for label, path in items:
        mark = "[green]✓[/green]" if path.exists() else "[dim]·[/dim]"
        console.print(f"  {mark} {label:<18} {path.name}")


if __name__ == "__main__":
    app()
