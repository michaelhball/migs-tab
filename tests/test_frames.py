"""Tests for frames.py — input validation for the zoom/crop options.

The ffmpeg invocation itself is a thin wrapper we don't unit-test; the
``migs-tab frame --zoom`` integration smoke-test on real audio covers the
happy path.
"""

from __future__ import annotations

import pytest

from migs_tab.frames import DEFAULT_FRETBOARD_CROP, extract_frame
from migs_tab.paths import VideoPaths


class TestDefaultFretboardCrop:
    def test_is_a_valid_rectangle(self):
        x0, y0, x1, y1 = DEFAULT_FRETBOARD_CROP
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0

    def test_keeps_only_upper_right_region(self):
        """Default crop should exclude the left edge (where the body sits)."""
        x0, _, _, _ = DEFAULT_FRETBOARD_CROP
        assert x0 > 0.0  # not from absolute left

    def test_keeps_only_middle_vertical_band(self):
        """Default crop should exclude the very top (face) and bottom (lap)."""
        _, y0, _, y1 = DEFAULT_FRETBOARD_CROP
        assert y0 > 0.0
        assert y1 < 1.0


class TestExtractFrameValidation:
    def test_negative_timestamp_rejected(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        # Create a fake video file so the "video not found" check passes.
        paths.video.write_bytes(b"")
        with pytest.raises(ValueError, match="timestamp_seconds"):
            extract_frame(paths, -1.0)

    def test_missing_video_raises(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Video not found"):
            extract_frame(paths, 1.0)

    def test_invalid_crop_rejected(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.video.write_bytes(b"")
        # x0 >= x1 (zero-width crop) — invalid.
        with pytest.raises(ValueError, match="invalid crop"):
            extract_frame(paths, 1.0, crop=(0.5, 0.0, 0.4, 1.0))

    def test_crop_outside_unit_range_rejected(self, tmp_path):
        paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
        paths.video.write_bytes(b"")
        with pytest.raises(ValueError, match="invalid crop"):
            extract_frame(paths, 1.0, crop=(0.0, 0.0, 1.5, 1.0))
