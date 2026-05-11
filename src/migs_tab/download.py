"""Download video, audio (as WAV), and captions for a YouTube URL via yt-dlp."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yt_dlp

from .paths import VideoPaths


def download(url: str, paths: VideoPaths, force: bool = False) -> VideoPaths:
    """Download video, extracted WAV audio, and English captions into the cache.

    Skips work if outputs already exist and ``force`` is False.
    """
    need_video = force or not paths.video.exists()
    need_audio = force or not paths.audio.exists()
    need_captions = force or not paths.captions_vtt.exists()

    if not (need_video or need_audio or need_captions):
        return paths

    # 1) Video (mp4)
    if need_video:
        _download_video(url, paths)

    # 2) Audio (wav, 44.1 kHz, mono — easier for Demucs/basic-pitch)
    if need_audio:
        _download_audio(url, paths)

    # 3) Captions + info.json
    if need_captions:
        _download_captions_and_info(url, paths)

    return paths


def _download_video(url: str, paths: VideoPaths) -> None:
    opts = {
        "outtmpl": str(paths.root / "video.%(ext)s"),
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # yt-dlp may produce video.mkv if merge fails or video.webm — normalize.
    if not paths.video.exists():
        for candidate in paths.root.glob("video.*"):
            if candidate.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                shutil.move(str(candidate), str(paths.video))
                break


def _download_audio(url: str, paths: VideoPaths) -> None:
    opts = {
        "outtmpl": str(paths.root / "audio.%(ext)s"),
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": [
            "-ar",
            "44100",
            "-ac",
            "2",
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def _download_captions_and_info(url: str, paths: VideoPaths) -> None:
    opts = {
        "outtmpl": str(paths.root / "captions.%(ext)s"),
        "skip_download": True,
        "writeinfojson": True,
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Persist a slimmed-down info.json (the file yt-dlp writes is huge).
    if info is not None:
        slim = {
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "channel": info.get("channel"),
            "duration": info.get("duration"),
            "upload_date": info.get("upload_date"),
            "description": info.get("description"),
            "webpage_url": info.get("webpage_url"),
        }
        paths.info_json.write_text(json.dumps(slim, indent=2))

    # Locate whichever VTT yt-dlp produced and normalize to captions.en.vtt
    if not paths.captions_vtt.exists():
        for candidate in paths.root.glob("captions.*.vtt"):
            shutil.move(str(candidate), str(paths.captions_vtt))
            break

    # Flatten captions to plain text for downstream LLM use.
    if paths.captions_vtt.exists():
        paths.captions_text.write_text(_vtt_to_plain_text(paths.captions_vtt))


def _vtt_to_plain_text(vtt_path: Path) -> str:
    """Strip VTT timing/tags down to readable transcript text, dedup repeats."""
    lines: list[str] = []
    prev = ""
    for raw in vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("Kind:")
            or line.startswith("Language:")
        ):
            continue
        if "-->" in line or line.startswith("NOTE"):
            continue
        # Strip <c> / <00:00:00.000> style inline tags.
        import re

        cleaned = re.sub(r"<[^>]+>", "", line).strip()
        if not cleaned or cleaned == prev:
            continue
        lines.append(cleaned)
        prev = cleaned
    return "\n".join(lines)
