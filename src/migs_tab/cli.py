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
from . import doctor as doctor_mod
from . import download as download_mod
from . import frames as frames_mod
from . import fret as fret_mod
from . import mt3 as mt3_mod
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


@app.command(name="transcribe-mt3")
def transcribe_mt3(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    force: bool = typer.Option(False, "--force"),
    variant: str = typer.Option(
        "YMT3+", "--variant", help="YourMT3+ model variant (see mt3.py for choices)"
    ),
    batch_size: int = typer.Option(2, "--batch-size"),
    on_mix: bool = typer.Option(
        False,
        "--on-mix",
        help="Transcribe the raw mix instead of the Demucs guitar stem "
        "(leaks instructor speech — debugging only)",
    ),
) -> None:
    """Transcribe via YourMT3+ — writes notes.mt3.mid + notes.mt3.json (MPS-aware).

    Defaults to the Demucs guitar stem when it exists, falling back to the raw
    mix only when no stem has been separated yet.
    """
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Transcribing[/bold] {paths.video_id} via YourMT3+ ({variant})")
    try:
        mt3_mod.transcribe(
            paths,
            force=force,
            variant=variant,
            batch_size=batch_size,
            audio_source=paths.audio if on_mix else None,
        )
    except mt3_mod.MT3NotInstalled as exc:
        console.print(f"[red]MT3 not installed:[/red] {exc}")
        raise typer.Exit(2) from exc
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
    backend: str = typer.Option(
        "mt3", "--backend", help="Which transcription to use: mt3 (default) or basic_pitch"
    ),
) -> None:
    """Assign string/fret positions to every note (Viterbi); write frets.json."""
    paths = _make_paths(url, cache_dir)
    console.print(f"[bold]Assigning frets[/bold] for {paths.video_id} (backend={backend})")
    fret_mod.assign_frets(paths, force=force, backend=backend)
    _print_status(paths)


@app.command()
def frame(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video ID"),
    timestamp: float = typer.Argument(..., help="Time in seconds to grab a frame"),
    label: str = typer.Option("", "--label", help="Suffix to append to the filename"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    zoom: bool = typer.Option(
        False,
        "--zoom",
        help="Crop to the fretboard area only (drops body + picking hand, ~3x zoom)",
    ),
    crop: str = typer.Option(
        "",
        "--crop",
        help="Custom crop as 'x0,y0,x1,y1' fractions 0..1 (e.g. '0.2,0.4,1.0,0.8'). Implies --zoom.",
    ),
) -> None:
    """Extract a single still frame from the video at the given timestamp."""
    paths = _make_paths(url, cache_dir)
    crop_tuple: tuple[float, float, float, float] | None = None
    if crop:
        parts = [float(x) for x in crop.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--crop must have four comma-separated fractions")
        crop_tuple = (parts[0], parts[1], parts[2], parts[3])
    out = frames_mod.extract_frame(
        paths, timestamp, label=label or None, zoom=zoom, crop=crop_tuple
    )
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
    mt3: bool = typer.Option(
        True,
        "--mt3/--no-mt3",
        help="Run YourMT3+ on MPS after Demucs, on the guitar stem (default: on)",
    ),
    mt3_on_mix: bool = typer.Option(
        False,
        "--mt3-on-mix",
        help="Feed MT3 the raw mix instead of the Demucs stem (leaks instructor "
        "speech as a 'Singing Voice' note channel — debugging only)",
    ),
    basic_pitch: bool = typer.Option(
        True,
        "--basic-pitch/--no-basic-pitch",
        help="Run basic-pitch (cheap, used for ornament hints; default: on)",
    ),
    backend: str = typer.Option(
        "mt3",
        "--backend",
        help="Which transcription drives the tab: mt3 (default) or basic_pitch",
    ),
    mt3_variant: str = typer.Option("YMT3+", "--mt3-variant"),
) -> None:
    """Pipeline: download → separate → transcribe (both backends) → structure → frets."""
    paths = _make_paths(url, cache_dir)
    console.rule(f"[bold cyan]migs-tab • {paths.video_id}")

    console.print("[bold]1/7[/bold] download")
    download_mod.download(url, paths, force=force)

    console.print("[bold]2/7[/bold] separate (Demucs)")
    separate_mod.separate(paths, force=force)

    if basic_pitch:
        console.print("[bold]3/7[/bold] transcribe (basic-pitch)")
        transcribe_mod.transcribe(paths, force=force)
    else:
        console.print("[bold]3/7[/bold] transcribe (basic-pitch) — [yellow]skipped[/yellow]")

    if mt3:
        # MT3 runs AFTER Demucs, on the clean guitar stem (MPS-backed driver).
        # Running it on the raw mix in parallel used to leak instructor speech
        # into the transcription as a 670-note "Singing Voice" channel —
        # correctness beats the lost parallelism. audio_source=None lets the
        # resolver prefer the stem and degrade to the mix if the stem is gone.
        mt3_source = paths.audio if mt3_on_mix else None
        if mt3_on_mix:
            source_label = "raw mix"
        elif paths.guitar_stem.exists():
            source_label = "guitar stem"
        else:
            source_label = "raw mix — stem missing"
        if force or not (paths.notes_mt3_midi.exists() and paths.notes_mt3_json.exists()):
            console.print(f"[bold]4/7[/bold] transcribe (YourMT3+ {mt3_variant}, {source_label})")
            try:
                mt3_mod.transcribe(paths, force=force, variant=mt3_variant, audio_source=mt3_source)
                console.print("[green]✓[/green] MT3 done")
            except mt3_mod.MT3NotInstalled as exc:
                console.print(f"[yellow]skipping MT3:[/yellow] {exc}")
            except subprocess.CalledProcessError as exc:
                console.print(
                    f"[red]MT3 exited with code {exc.returncode}[/red] — "
                    "continuing with basic-pitch only"
                )
        else:
            import json as _json

            cached_source = None
            try:
                provenance = _json.loads(paths.notes_mt3_json.read_text()).get("provenance") or {}
                cached_source = provenance.get("audio_source")
            except (OSError, ValueError):
                pass
            suffix = f" (source: {cached_source})" if cached_source else ""
            console.print(f"[bold]4/7[/bold] transcribe (YourMT3+) — cached, skipping{suffix}")
    else:
        console.print("[bold]4/7[/bold] transcribe (YourMT3+) — [yellow]skipped[/yellow]")

    console.print("[bold]5/7[/bold] structure (librosa)")
    structure_mod.analyze_structure(paths, force=force)

    console.print("[bold]6/7[/bold] tuning")
    tuning_mod.detect_tuning(paths, force=force)

    console.print(f"[bold]7/7[/bold] frets (Viterbi, backend={backend})")
    fret_mod.assign_frets(paths, force=force, backend=backend)

    console.rule("[bold green]done — LLM steps now run via the /migs-tab skill")
    _print_status(paths)


@app.command()
def doctor() -> None:
    """Preflight check — verify Python, ffmpeg, deps, cache state."""
    console.print("[bold]migs-tab doctor[/bold]\n")
    results = doctor_mod.run_checks()
    any_failed = False
    for r in results:
        mark = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        console.print(f"  {mark}  {r.name:<14} {r.detail}")
        if not r.ok:
            any_failed = True
    console.print()
    if any_failed:
        console.print("[yellow]Some checks failed — see ✗ entries above.[/yellow]")
        raise typer.Exit(code=1)
    console.print("[green]All checks passed.[/green]")


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
