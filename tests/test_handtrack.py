"""Tests for handtrack.py — role assignment and frame-stream plumbing.

The MediaPipe detector itself is not exercised here (it needs the downloaded
model); the pure functions around it are.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from migs_tab.handtrack import (
    FrameRecord,
    HandDetection,
    _two_means_threshold,
    assign_roles,
    ffmpeg_frame_command,
    iter_frames,
    probe_video_size,
)


def _hand(wrist_x: float, label: str = "Left") -> HandDetection:
    lm = [[wrist_x, 0.8]] + [[wrist_x - 0.01 * i, 0.75] for i in range(1, 21)]
    return HandDetection(label=label, score=0.99, landmarks=lm)


class TestAssignRoles:
    def test_two_hand_frames_fix_roles_by_position(self):
        recs = [
            FrameRecord(0, 0.0, [_hand(0.17), _hand(0.70)]),
            FrameRecord(1, 0.2, [_hand(0.71), _hand(0.16)]),
        ]
        summary = assign_roles(recs)
        assert summary["method"] == "two-hand-frames"
        for r in recs:
            roles = sorted(h.role for h in r.hands)
            assert roles == ["fretting", "picking"]
            assert r.fretting().wrist_x > 0.5

    def test_single_hand_frames_classified_by_threshold(self):
        recs = [
            FrameRecord(0, 0.0, [_hand(0.17), _hand(0.70)]),
            FrameRecord(1, 0.2, [_hand(0.69)]),  # fretting hand alone
            FrameRecord(2, 0.4, [_hand(0.20)]),  # picking hand alone
            FrameRecord(3, 0.6, []),
        ]
        summary = assign_roles(recs)
        assert recs[1].hands[0].role == "fretting"
        assert recs[2].hands[0].role == "picking"
        assert recs[3].fretting() is None
        assert summary["frames_with_fretting_hand"] == 2
        assert summary["frames_total"] == 4

    def test_mirrored_footage_flips_the_rule(self):
        recs = [FrameRecord(0, 0.0, [_hand(0.17), _hand(0.70)])]
        assign_roles(recs, headstock_right=False)
        assert recs[0].fretting().wrist_x == pytest.approx(0.17)

    def test_no_two_hand_frames_uses_two_means(self):
        # Fretting hand mostly alone around 0.9, two stray picking detections at 0.48.
        recs = [FrameRecord(i, i * 0.2, [_hand(0.90 + 0.005 * (i % 3))]) for i in range(10)]
        recs += [FrameRecord(10, 2.0, [_hand(0.48)]), FrameRecord(11, 2.2, [_hand(0.47)])]
        summary = assign_roles(recs)
        assert summary["method"] == "two-means"
        assert recs[0].hands[0].role == "fretting"
        assert recs[10].hands[0].role == "picking"

    def test_single_population_marks_everything_fretting(self):
        recs = [FrameRecord(i, i * 0.2, [_hand(0.70 + 0.01 * (i % 4))]) for i in range(8)]
        summary = assign_roles(recs)
        assert summary["method"] == "single-population"
        assert all(r.hands[0].role == "fretting" for r in recs)


class TestTwoMeans:
    def test_too_few(self):
        assert _two_means_threshold([0.1, 0.9], 1.0) == (None, "too-few-detections")

    def test_separates_clear_clusters(self):
        thr, method = _two_means_threshold([0.1, 0.12, 0.11, 0.8, 0.82, 0.81], 1.0)
        assert method == "two-means"
        assert 0.3 < thr < 0.6


class TestFrameStream:
    def test_ffmpeg_command_requests_rgb24_at_fps(self, tmp_path):
        cmd = ffmpeg_frame_command(tmp_path / "v.mp4", 5.0)
        assert cmd[0] == "ffmpeg"
        assert "fps=5.0" in cmd
        assert "rgb24" in cmd
        assert cmd[-1] == "-"

    def test_iter_frames_decodes_a_synthetic_video(self, tmp_path):
        video = tmp_path / "v.mp4"
        # 2 s of a 64x48 test pattern at 10 fps -> sampled at 5 fps = 10 frames.
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=10:duration=2",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        assert probe_video_size(video) == (64, 48)
        frames = list(iter_frames(video, 5.0))
        assert len(frames) == 10
        idx, t, arr = frames[3]
        assert idx == 3
        assert t == pytest.approx(0.6)
        assert arr.shape == (48, 64, 3)
        assert arr.dtype == np.uint8
