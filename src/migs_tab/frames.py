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


def extract_frame(
    paths: VideoPaths,
    timestamp_seconds: float,
    out_dir: Path | None = None,
    label: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract a single video frame at the given timestamp.

    The file is named ``t<seconds>_<label>.jpg`` (label optional) and lives
    in ``cache/<id>/frames/`` unless ``out_dir`` overrides.
    """
    if not paths.video.exists():
        raise FileNotFoundError(f"Video not found at {paths.video}")
    if timestamp_seconds < 0:
        raise ValueError("timestamp_seconds must be >= 0")

    target_dir = out_dir if out_dir is not None else paths.frames_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = f"{timestamp_seconds:08.3f}".replace(".", "_")
    suffix = f"_{label}" if label else ""
    out_path = target_dir / f"t{stamp}{suffix}.jpg"

    if out_path.exists() and not overwrite:
        return out_path

    cmd = [
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
        str(out_path),
    ]
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
