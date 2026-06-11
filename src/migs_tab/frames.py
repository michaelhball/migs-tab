"""Phase 3.5 helper: extract still frames from the video at specified
onset timestamps so the Claude skill can read them and disambiguate fret
positions from visible hand geometry.

We deliberately keep this small and explicit. Each call writes to
``cache/<id>/frames/<subdir>/`` so the skill can group frames per
ambiguous cluster, and a hard ``--max-frames`` ceiling protects the
user's Claude subscription quota from accidental batch reads.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .paths import VideoPaths

# Default fretboard crop region as fractions of (width, height). Tuned for
# the typical front-facing seated-instructor framing (Shutup&Play, Marty
# Music, etc.) where the body occupies the lower-left and the neck extends
# upper-right. Crops out the body+picking-hand and zooms ~3x on the
# fretboard, making individual fret dots and finger placements legible.
#
# Override via the CLI when a video has unusual framing. Encoded as
# (x_start, y_start, x_end, y_end) where each is a fraction in [0, 1].
DEFAULT_FRETBOARD_CROP = (0.20, 0.40, 1.00, 0.80)


def extract_frame(
    paths: VideoPaths,
    timestamp_seconds: float,
    out_dir: Path | None = None,
    label: str | None = None,
    overwrite: bool = False,
    zoom: bool = False,
    crop: tuple[float, float, float, float] | None = None,
) -> Path:
    """Extract a single video frame at the given timestamp.

    The file is named ``t<seconds>_<label>.jpg`` (label optional) and lives
    in ``cache/<id>/frames/`` unless ``out_dir`` overrides.

    Set ``zoom=True`` to apply ``DEFAULT_FRETBOARD_CROP`` so the fretboard
    fills the frame — much easier to read individual frets. Pass an explicit
    ``crop=(x0, y0, x1, y1)`` (each a 0..1 fraction) to override the default.
    """
    if not paths.video.exists():
        raise FileNotFoundError(f"Video not found at {paths.video}")
    if timestamp_seconds < 0:
        raise ValueError("timestamp_seconds must be >= 0")

    target_dir = out_dir if out_dir is not None else paths.frames_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = f"{timestamp_seconds:08.3f}".replace(".", "_")
    suffix = f"_{label}" if label else ""
    if zoom or crop is not None:
        suffix += "_zoom"
    out_path = target_dir / f"t{stamp}{suffix}.jpg"

    if out_path.exists() and not overwrite:
        return out_path

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(paths.video),
        "-frames:v",
        "1",
        "-q:v",
        "3",  # JPEG quality 3 ~ visually lossless, ~50-150 KB per frame
    ]
    if zoom or crop is not None:
        x0, y0, x1, y1 = crop if crop is not None else DEFAULT_FRETBOARD_CROP
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"invalid crop fractions: {(x0, y0, x1, y1)!r}")
        # ffmpeg crop=w:h:x:y using fractional expressions on input dims.
        cmd.extend(
            [
                "-vf",
                f"crop=in_w*{x1 - x0:.4f}:in_h*{y1 - y0:.4f}:in_w*{x0:.4f}:in_h*{y0:.4f}",
            ]
        )
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)
    return out_path


def extract_frames_for_clusters(
    paths: VideoPaths,
    cluster_ids: list[int],
    max_frames: int,
    subdir: str = "ambiguous",
) -> list[dict]:
    """Extract one frame at each cluster's onset. Bounded by ``max_frames``."""
    if not paths.frets_json.exists():
        raise FileNotFoundError(f"frets.json not found at {paths.frets_json}; run frets first.")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    data = json.loads(paths.frets_json.read_text())
    clusters = {c["cluster_id"]: c for c in data["clusters"]}

    requested = list(cluster_ids)
    selected = requested[:max_frames]
    skipped = len(requested) - len(selected)

    out_dir = paths.frames_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for cid in selected:
        if cid not in clusters:
            records.append({"cluster_id": cid, "error": "no such cluster"})
            continue
        c = clusters[cid]
        frame_path = extract_frame(paths, c["onset"], out_dir=out_dir, label=f"cluster{cid}")
        records.append(
            {
                "cluster_id": cid,
                "onset": c["onset"],
                "frame_path": str(frame_path),
                "current_choice": [
                    {"string": a["string"], "fret": a["fret"]}
                    for a in [n for n in data["notes"] if n["cluster_id"] == cid]
                ],
                "alternatives": c.get("alternatives", []),
            }
        )

    return [
        {
            "extracted_count": len(records),
            "skipped_due_to_cap": skipped,
            "max_frames": max_frames,
            "cluster_records": records,
        }
    ][0]
