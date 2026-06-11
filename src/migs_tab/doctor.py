"""Preflight check for migs-tab. Reports whether each external dependency
is present and the project's cache/output state — the first thing to run
when something isn't working as expected.
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT, VideoPaths


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


_MT3_DIR = PROJECT_ROOT / "third_party" / "YourMT3"
_MT3_DEFAULT_CKPT = (
    _MT3_DIR
    / "amt"
    / "logs"
    / "2024"
    / "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72"
    / "checkpoints"
    / "model.ckpt"
)
_MT3_VENV_PYTHON = _MT3_DIR / ".venv" / "bin" / "python"
# LFS pointer files are ~130 bytes; the real checkpoint is >100 MB. Any value
# in between catches "you forgot to git lfs pull" without false negatives.
_MT3_CKPT_MIN_BYTES = 10 * 1024 * 1024


def _check_mt3_install() -> CheckResult:
    """Optional: YourMT3+ vendored install (driver + venv + lightest checkpoint)."""
    if not _MT3_DIR.exists():
        return CheckResult("mt3 (optional)", True, "not installed — basic-pitch is the default")
    missing: list[str] = []
    if not (_MT3_DIR / "migs_driver.py").exists():
        missing.append("driver script")
    if not _MT3_VENV_PYTHON.exists():
        missing.append(f"venv at {_MT3_VENV_PYTHON.parent.parent}")
    if not _MT3_DEFAULT_CKPT.exists():
        missing.append("YMT3+ checkpoint")
    elif _MT3_DEFAULT_CKPT.stat().st_size < _MT3_CKPT_MIN_BYTES:
        missing.append("YMT3+ checkpoint (LFS pointer only — run `git lfs pull`)")
    if missing:
        return CheckResult("mt3 (optional)", False, "missing: " + ", ".join(missing))
    size_mb = _MT3_DEFAULT_CKPT.stat().st_size / (1024 * 1024)
    return CheckResult("mt3 (optional)", True, f"venv + YMT3+ checkpoint ready ({size_mb:.0f} MB)")


# Canonical, version-tracked copy of the YourMT3 driver (see
# scripts/yourmt3/SETUP.md). third_party/YourMT3/ is gitignored, so edits to
# the deployed copy silently drift from the tracked one — hash-compare them.
_DRIVER_CANONICAL = PROJECT_ROOT / "scripts" / "yourmt3" / "migs_driver.py"
_DRIVER_DEPLOYED = _MT3_DIR / "migs_driver.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ast_dump(path: Path) -> str | None:
    """Canonical AST dump of a Python source file; None when unparseable."""
    try:
        return ast.dump(ast.parse(path.read_text()))
    except (SyntaxError, ValueError, OSError):
        return None


def _check_driver_sync() -> CheckResult:
    """Warn when the deployed MT3 driver drifts from the tracked copy.

    Formatting-only drift (byte-different but AST-identical, e.g. a ruff
    re-format of the tracked copy that was never re-deployed) passes with a
    reconcile nudge instead of failing: doctor exits 1 on any failed check,
    and a cosmetic diff must not break scripts gating on that exit code.
    Behavioral (AST-visible) drift still fails hard.
    """
    if not _DRIVER_DEPLOYED.exists():
        return CheckResult("mt3 driver", True, "third_party copy not deployed — nothing to compare")
    if not _DRIVER_CANONICAL.exists():
        return CheckResult("mt3 driver", False, f"canonical copy missing at {_DRIVER_CANONICAL}")
    canonical, deployed = _sha256(_DRIVER_CANONICAL), _sha256(_DRIVER_DEPLOYED)
    if canonical == deployed:
        return CheckResult(
            "mt3 driver", True, f"deployed copy matches scripts/yourmt3 (sha256 {canonical[:12]})"
        )
    ast_canonical = _ast_dump(_DRIVER_CANONICAL)
    if ast_canonical is not None and ast_canonical == _ast_dump(_DRIVER_DEPLOYED):
        return CheckResult(
            "mt3 driver",
            True,
            f"formatting-only drift (ASTs identical): deployed sha256 {deployed[:12]} != "
            f"tracked {canonical[:12]} — "
            "`cp scripts/yourmt3/migs_driver.py third_party/YourMT3/` to silence",
        )
    return CheckResult(
        "mt3 driver",
        False,
        f"DRIFT: {_DRIVER_DEPLOYED} (sha256 {deployed[:12]}) != tracked "
        f"{_DRIVER_CANONICAL} (sha256 {canonical[:12]}) — reconcile, then "
        "`cp scripts/yourmt3/migs_driver.py third_party/YourMT3/`",
    )


def _check_verification() -> CheckResult:
    """Report whether each fret-assigned video has a fresh verification.json
    (fresh = at least as new as EVERY verify input: frets, both note files,
    sections, tuning, the stem — verify.verification_input_paths)."""
    # Lazy import: verify pulls librosa via salience, and doctor must still
    # run (and report the broken package) when that import chain is dead.
    try:
        from .verify import verification_input_paths
    except ImportError as e:
        return CheckResult("verification", False, f"verify module unimportable: {e!r}")
    if not DEFAULT_CACHE_DIR.exists():
        return CheckResult("verification", True, "no cache yet")
    fresh: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    for d in sorted(DEFAULT_CACHE_DIR.iterdir()):
        frets = d / "frets.json"
        if not d.is_dir() or not frets.exists():
            continue
        verification = d / "verification.json"
        inputs = verification_input_paths(VideoPaths(d.name, cache_dir=DEFAULT_CACHE_DIR))
        newest_input = max((p.stat().st_mtime for p in inputs if p.exists()), default=0.0)
        if not verification.exists():
            missing.append(d.name)
        elif verification.stat().st_mtime >= newest_input:
            fresh.append(d.name)
        else:
            stale.append(d.name)
    total = len(fresh) + len(stale) + len(missing)
    if not total:
        return CheckResult("verification", True, "no fret-assigned videos yet")
    detail = f"{len(fresh)}/{total} video(s) fresh"
    if stale:
        detail += (
            f"; STALE (older than a verify input): {', '.join(stale)} — re-run `migs-tab verify`"
        )
    if missing:
        detail += f"; never verified: {', '.join(missing)}"
    return CheckResult("verification", not stale, detail)


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
        _check_mt3_install(),
        _check_driver_sync(),
        _check_cache_dir(),
        _check_output_dir(),
        _check_verification(),
        _check_disk_space(),
    ]
