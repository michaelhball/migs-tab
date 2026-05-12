"""Cache and output directory helpers, keyed by YouTube video ID."""

from __future__ import annotations

import json
import re
import unicodedata
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

    @property
    def tuning_json(self) -> Path:
        return self.root / "tuning.json"

    def output_dir(self, output_root: Path) -> Path:
        """Return the output dir for this video. Uses a slugified version of
        the video's title from info.json so files are easy to find; falls
        back to the 11-char video_id if no title is cached."""
        slug = self._title_slug()
        d = output_root / slug
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _title_slug(self) -> str:
        """Read info.json and produce a filesystem-safe slug like
        `how-to-play-angie-the-rolling-stones-acoustic` (truncated to a
        reasonable length). Falls back to the bare video ID."""
        if not self.info_json.exists():
            return self.video_id
        try:
            info = json.loads(self.info_json.read_text())
        except (json.JSONDecodeError, OSError):
            return self.video_id
        title = (info.get("title") or "").strip()
        if not title:
            return self.video_id
        slug = slugify_title(title)
        if not slug:
            return self.video_id
        # Suffix with a short ID hash so two videos with the same title still
        # land in different dirs (rare but possible across re-uploads).
        return f"{slug}-{self.video_id}"


def slugify_title(title: str, max_length: int = 60) -> str:
    """Convert a video title into a filesystem-friendly slug.

    Lowercase, ASCII-only, words separated by single dashes, truncated to
    ``max_length`` (defaults to 60 to keep paths short).
    """
    # Decompose accents + drop non-ASCII bytes.
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    # Lowercase + replace anything that isn't [a-z0-9] with dashes.
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug
