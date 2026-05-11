"""Cache and output directory helpers, keyed by YouTube video ID."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_OUTPUT_DIR",
    "PROJECT_ROOT",
    "VideoPaths",
    "extract_video_id",
]

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Pull the 11-char YouTube video ID from a URL, or accept a bare ID."""
    if _VIDEO_ID_RE.match(url_or_id):
        return url_or_id

    parsed = urlparse(url_or_id)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be"}:
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(candidate):
            return candidate

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and _VIDEO_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        # /embed/<id> or /shorts/<id>
        parts = [p for p in parsed.path.split("/") if p]
        for i, part in enumerate(parts):
            if part in {"embed", "shorts", "live"} and i + 1 < len(parts):
                candidate = parts[i + 1]
                if _VIDEO_ID_RE.match(candidate):
                    return candidate

    raise ValueError(f"Could not extract YouTube video ID from: {url_or_id!r}")


class VideoPaths:
    """All on-disk paths for a single video's cached artifacts."""

    def __init__(self, video_id: str, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.video_id = video_id
        self.root = cache_dir / video_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def video(self) -> Path:
        return self.root / "video.mp4"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def info_json(self) -> Path:
        return self.root / "info.json"

    @property
    def captions_vtt(self) -> Path:
        return self.root / "captions.en.vtt"

    @property
    def captions_text(self) -> Path:
        return self.root / "captions.txt"

    @property
    def stems_dir(self) -> Path:
        return self.root / "stems"

    @property
    def guitar_stem(self) -> Path:
        return self.stems_dir / "other.wav"

    @property
    def notes_midi(self) -> Path:
        return self.root / "notes.mid"

    @property
    def notes_json(self) -> Path:
        return self.root / "notes.json"

    @property
    def tips_md(self) -> Path:
        return self.root / "tips.md"

    @property
    def structure_json(self) -> Path:
        return self.root / "structure.json"

    @property
    def sections_json(self) -> Path:
        return self.root / "sections.json"

    @property
    def frets_json(self) -> Path:
        return self.root / "frets.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def frets_overrides_json(self) -> Path:
        return self.root / "frets.overrides.json"

    def output_dir(self, output_root: Path) -> Path:
        d = output_root / self.video_id
        d.mkdir(parents=True, exist_ok=True)
        return d
