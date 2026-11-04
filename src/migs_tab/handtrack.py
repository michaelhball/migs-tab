"""Fretting-hand tracking from the tutorial video.

Runs MediaPipe's hand-landmark model over the whole video at a fixed frame
rate and writes ``cache/<id>/hand_track.json``: for every sampled frame, the
21 landmarks (normalized x, y in [0, 1]) of each detected hand, plus a role
per hand ("fretting" or "picking") decided from the hand's position along the
neck.

Why this exists: string and fret cannot be recovered from pitch alone (a
pitch is playable in up to five places), and audio-only assignment tops out
around 70-75% string accuracy even for trained models. The player's hand is
on camera the whole time in a tutorial, and its position along the neck
settles that ambiguity for most notes. This module only produces the raw
track; ``fretboard.py`` maps pixel positions to fret numbers.

Role assignment assumes a right-handed player filmed with the headstock on
the right of the frame (the common tutorial framing), so the fretting hand
is the one further right. Pass ``headstock_right=False`` for mirrored
footage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .paths import DEFAULT_CACHE_DIR, VideoPaths

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = DEFAULT_CACHE_DIR / "_models" / "hand_landmarker.task"
_HAND_MODEL_MIN_BYTES = 1_000_000

DEFAULT_FPS = 5.0
# MediaPipe landmark indices (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
WRIST = 0
INDEX_MCP = 5
FINGERTIPS = (8, 12, 16, 20)  # index, middle, ring, pinky
# When the two hands never appear together, a 2-means split on wrist x must
# separate by at least this much (fraction of frame width) to count as two
# populations; otherwise every detection is treated as the fretting hand.
_MIN_ROLE_SEPARATION = 0.15


@dataclass
class HandDetection:
    label: str  # MediaPipe handedness label; informational only
    score: float
    landmarks: list[list[float]]  # 21 × [x, y], normalized to the frame
    role: str = "unknown"  # "fretting" | "picking" | "unknown"

    @property
    def wrist_x(self) -> float:
        return self.landmarks[WRIST][0]


@dataclass
class FrameRecord:
    index: int
    t: float
    hands: list[HandDetection]

    def fretting(self) -> HandDetection | None:
        for h in self.hands:
            if h.role == "fretting":
                return h
        return None


# ---------------------------------------------------------------------------
# Model file
# ---------------------------------------------------------------------------


def ensure_hand_model(path: Path = HAND_MODEL_PATH) -> Path:
    """Download the MediaPipe hand-landmark model on first use."""
    if path.exists() and path.stat().st_size >= _HAND_MODEL_MIN_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".task.part")
    with urllib.request.urlopen(HAND_MODEL_URL, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    if tmp.stat().st_size < _HAND_MODEL_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"hand model download looks truncated ({tmp.stat().st_size} bytes)")
    tmp.replace(path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------


def probe_video_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def ffmpeg_frame_command(video: Path, fps: float) -> list[str]:
    """ffmpeg command that streams RGB24 frames at ``fps`` to stdout."""
    return [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def iter_frames(video: Path, fps: float):
    """Yield (index, t_seconds, rgb_array) for every sampled frame."""
    w, h = probe_video_size(video)
    nbytes = w * h * 3
    proc = subprocess.Popen(ffmpeg_frame_command(video, fps), stdout=subprocess.PIPE)
    assert proc.stdout is not None
    i = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            yield i, i / fps, frame
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _make_landmarker(model_path: Path):
    # Imported here: mediapipe is a heavy optional dependency, and importing it
    # at module load would slow every CLI invocation that never touches video.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts), mp


def detect_hands(video: Path, fps: float, model_path: Path) -> list[FrameRecord]:
    landmarker, mp = _make_landmarker(model_path)
    records: list[FrameRecord] = []
    for i, t, frame in iter_frames(video, fps):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = landmarker.detect_for_video(image, int(round(t * 1000)))
        hands: list[HandDetection] = []
        for lm, handed in zip(result.hand_landmarks, result.handedness, strict=False):
            hands.append(
                HandDetection(
                    label=handed[0].category_name,
                    score=round(float(handed[0].score), 3),
                    landmarks=[[round(p.x, 4), round(p.y, 4)] for p in lm],
                )
            )
        records.append(FrameRecord(index=i, t=round(t, 3), hands=hands))
    return records


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------


def assign_roles(records: list[FrameRecord], headstock_right: bool = True) -> dict:
    """Label each detection "fretting" or "picking" from its wrist x.

    Frames with two hands fix the two populations directly: the hand nearer
    the headstock is the fretting hand. Frames with one hand are classified
    by the midpoint between the two population medians. If no frame ever shows
    both hands, a 1-D 2-means split on wrist x is used; if that split is
    narrower than ``_MIN_ROLE_SEPARATION`` every detection is called fretting.
    Returns a summary dict for the JSON header.
    """
    sign = 1.0 if headstock_right else -1.0
    fret_xs: list[float] = []
    pick_xs: list[float] = []
    for rec in records:
        if len(rec.hands) == 2:
            a, b = rec.hands
            fret, pick = (a, b) if sign * a.wrist_x > sign * b.wrist_x else (b, a)
            fret.role, pick.role = "fretting", "picking"
            fret_xs.append(fret.wrist_x)
            pick_xs.append(pick.wrist_x)

    if fret_xs and pick_xs:
        threshold = (statistics.median(fret_xs) + statistics.median(pick_xs)) / 2
        method = "two-hand-frames"
    else:
        xs = sorted(h.wrist_x for r in records for h in r.hands)
        threshold, method = _two_means_threshold(xs, sign)

    for rec in records:
        if len(rec.hands) == 1:
            h = rec.hands[0]
            if threshold is None:
                h.role = "fretting"
            else:
                h.role = "fretting" if sign * h.wrist_x > sign * threshold else "picking"

    n_fret = sum(1 for r in records if r.fretting() is not None)
    return {
        "method": method,
        "threshold_x": None if threshold is None else round(threshold, 4),
        "two_hand_frames": len(fret_xs),
        "frames_with_fretting_hand": n_fret,
        "frames_total": len(records),
    }


def _two_means_threshold(xs: list[float], sign: float) -> tuple[float | None, str]:
    if len(xs) < 4:
        return None, "too-few-detections"
    lo, hi = min(xs), max(xs)
    c1, c2 = lo, hi
    for _ in range(50):
        g1 = [x for x in xs if abs(x - c1) <= abs(x - c2)]
        g2 = [x for x in xs if abs(x - c1) > abs(x - c2)]
        if not g1 or not g2:
            break
        n1, n2 = statistics.mean(g1), statistics.mean(g2)
        if abs(n1 - c1) < 1e-6 and abs(n2 - c2) < 1e-6:
            break
        c1, c2 = n1, n2
    if abs(c2 - c1) < _MIN_ROLE_SEPARATION:
        return None, "single-population"
    return (c1 + c2) / 2, "two-means"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def track_hands(
    paths: VideoPaths,
    fps: float = DEFAULT_FPS,
    force: bool = False,
    headstock_right: bool = True,
    model_path: Path = HAND_MODEL_PATH,
) -> Path:
    """Run the tracker over ``paths.video`` and write ``paths.hand_track_json``."""
    out = paths.hand_track_json
    if out.exists() and not force:
        return out
    if not paths.video.exists():
        raise FileNotFoundError(f"video not found at {paths.video}; run download first")
    model = ensure_hand_model(model_path)
    w, h = probe_video_size(paths.video)
    t0 = time.monotonic()
    records = detect_hands(paths.video, fps, model)
    roles = assign_roles(records, headstock_right=headstock_right)
    payload = {
        "video_id": paths.video_id,
        "fps": fps,
        "frame_width": w,
        "frame_height": h,
        "headstock_right": headstock_right,
        "model": {"path": str(model), "sha256": _sha256(model)},
        "elapsed_s": round(time.monotonic() - t0, 1),
        "roles": roles,
        "frames": [asdict(r) for r in records],
    }
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return out


def load_hand_track(paths: VideoPaths) -> dict:
    return json.loads(paths.hand_track_json.read_text())
