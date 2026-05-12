"""Tests for chord_shapes.py — frame picker logic, sharpness scoring."""

from __future__ import annotations

import numpy as np
from PIL import Image

from migs_tab.chord_shapes import _evenly_sampled_spans, _frame_sharpness, _safe_label


class TestSafeLabel:
    def test_basic(self):
        assert _safe_label("Am") == "Am"

    def test_sharp(self):
        assert _safe_label("F#") == "Fsharp"

    def test_slash(self):
        assert _safe_label("G/B") == "G_over_B"

    def test_combined(self):
        assert _safe_label("D#m/F") == "Dsharpm_over_F"


class TestEvenlySampledSpans:
    def test_short_list_returned_intact(self):
        spans = [{"start": 0.0}, {"start": 5.0}]
        assert _evenly_sampled_spans(spans, 5) == spans

    def test_picks_evenly_from_longer_list(self):
        spans = [{"start": float(i)} for i in range(20)]
        picked = _evenly_sampled_spans(spans, 5)
        assert len(picked) == 5
        # First and last should be included.
        assert picked[0]["start"] == 0.0
        assert picked[-1]["start"] == 19.0

    def test_dedupes_consecutive(self):
        # If indices land on the same span, dedup so we don't have duplicates.
        spans = [{"start": 0.0}, {"start": 1.0}]
        picked = _evenly_sampled_spans(spans, 3)
        # 2 spans, sampling 3 → indices [0, 1, 1] → dedup to [0, 1].
        assert len(picked) == 2


class TestFrameSharpness:
    def _make_sharp_image(self, tmp_path):
        # An image with sharp edges (alternating black/white squares).
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[::4, :] = 255  # horizontal stripes
        path = tmp_path / "sharp.jpg"
        Image.fromarray(arr).save(path, quality=95)
        return path

    def _make_blurry_image(self, tmp_path):
        # A uniform gray image — no edges.
        arr = np.full((100, 100), 128, dtype=np.uint8)
        path = tmp_path / "blurry.jpg"
        Image.fromarray(arr).save(path, quality=95)
        return path

    def test_sharp_higher_than_blurry(self, tmp_path):
        sharp_path = self._make_sharp_image(tmp_path)
        blurry_path = self._make_blurry_image(tmp_path)
        sharp_score = _frame_sharpness(sharp_path)
        blurry_score = _frame_sharpness(blurry_path)
        assert sharp_score > blurry_score
