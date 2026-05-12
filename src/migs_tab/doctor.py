"""Preflight check for migs-tab. Reports whether each external dependency
is present and the project's cache/output state — the first thing to run
when something isn't working as expected.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_python() -> CheckResult:
    major, minor = sys.version_info[:2]
    ok = (major, minor) == (3, 11)
    detail = f"Python {major}.{minor}.{sys.version_info.micro}"
    if not ok:
        detail += "  (expected 3.11 — pyproject.toml pins this)"
    return CheckResult("python", ok, detail)


def _check_binary(name: str, version_flag: str = "--version") -> CheckResult:
    path = shutil.which(name)
    if not path:
        return CheckResult(name, False, "not found in PATH")
    try:
        proc = subprocess.run(
            [path, version_flag],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = (
            (proc.stdout or proc.stderr).splitlines()[0] if (proc.stdout or proc.stderr) else "?"
        )
        return CheckResult(name, True, f"{path}  ({first_line.strip()})")
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult(name, False, f"failed to run: {e!r}")


def _check_python_package(import_name: str) -> CheckResult:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        return CheckResult(import_name, True, f"version {version}")
    except ImportError as e:
        return CheckResult(import_name, False, f"import failed: {e!r}")


def _check_cache_dir() -> CheckResult:
    if not DEFAULT_CACHE_DIR.exists():
        return CheckResult(
            "cache dir", True, f"{DEFAULT_CACHE_DIR} (empty / will be created on first run)"
        )
    video_dirs = [d for d in DEFAULT_CACHE_DIR.iterdir() if d.is_dir()]
    size = _dir_size(DEFAULT_CACHE_DIR)
    return CheckResult(
        "cache dir",
        True,
        f"{DEFAULT_CACHE_DIR}  ({len(video_dirs)} video(s), {_human_bytes(size)})",
    )


def _check_output_dir() -> CheckResult:
    if not DEFAULT_OUTPUT_DIR.exists():
        return CheckResult("output dir", True, f"{DEFAULT_OUTPUT_DIR} (no renders yet)")
    out_dirs = [d for d in DEFAULT_OUTPUT_DIR.iterdir() if d.is_dir()]
    return CheckResult(
        "output dir", True, f"{DEFAULT_OUTPUT_DIR}  ({len(out_dirs)} rendered song(s))"
    )


def _check_disk_space() -> CheckResult:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    # Demucs + cache for a single video needs ~1 GB; warn if < 3 GB free.
    ok = free_gb >= 3.0
    detail = f"{free_gb:.1f} GB free at {PROJECT_ROOT}"
    if not ok:
        detail += "  (less than 3 GB free — Demucs runs may fail)"
    return CheckResult("disk space", ok, detail)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def run_checks() -> list[CheckResult]:
    """Run every check and return the results in order."""
    return [
        _check_python(),
        _check_binary("ffmpeg", "-version"),
        _check_binary("ffprobe", "-version"),
        _check_python_package("librosa"),
        _check_python_package("torch"),
        _check_python_package("demucs"),
        _check_python_package("basic_pitch"),
        _check_python_package("yt_dlp"),
        _check_python_package("PIL"),
        _check_python_package("scipy"),
        _check_python_package("numpy"),
        _check_cache_dir(),
        _check_output_dir(),
        _check_disk_space(),
    ]
