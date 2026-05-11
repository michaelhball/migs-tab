"""Run Demucs to isolate the guitar stem from the downloaded audio.

We use the 4-source ``htdemucs`` model and take the ``other`` stem, which is
the closest proxy to acoustic guitar after vocals / drums / bass are removed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from .paths import VideoPaths


def separate(
    paths: VideoPaths,
    model: str = "htdemucs",
    force: bool = False,
    audio_name: str = "audio",
) -> VideoPaths:
    target_audio = paths.root / f"{audio_name}.wav"
    target_stem = (
        paths.guitar_stem if audio_name == "audio" else paths.stems_dir / f"other.{audio_name}.wav"
    )

    if target_stem.exists() and not force:
        return paths

    if not target_audio.exists():
        raise FileNotFoundError(
            f"Audio not found for {paths.video_id} at {target_audio}. "
            "Run the download step (and clip step, if using a clip name) first."
        )

    paths.stems_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "other",
        "-n",
        model,
        "-o",
        str(paths.root / "_demucs_out"),
        str(target_audio),
    ]
    subprocess.run(cmd, check=True)

    track_name = target_audio.stem
    demucs_track_dir = paths.root / "_demucs_out" / model / track_name
    src = demucs_track_dir / "other.wav"
    if not src.exists():
        raise RuntimeError(f"Expected Demucs output at {src} but it does not exist")

    shutil.move(str(src), str(target_stem))

    # Optionally also keep the 'no_other' (everything-but-guitar) stem for debugging.
    no_other = demucs_track_dir / "no_other.wav"
    if no_other.exists():
        shutil.move(str(no_other), str(paths.stems_dir / "no_other.wav"))

    # Clean up demucs temp tree.
    shutil.rmtree(paths.root / "_demucs_out", ignore_errors=True)

    return paths
