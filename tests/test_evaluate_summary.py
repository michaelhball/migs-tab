"""evaluate_video: unverified passages are reported but excluded from the summary."""

from __future__ import annotations

import json

from migs_tab.evaluate import evaluate_video
from migs_tab.paths import VideoPaths


def test_unverified_passages_stay_out_of_the_summary(tmp_path):
    paths = VideoPaths("aaa11111111", cache_dir=tmp_path)
    paths.frets_json.write_text(
        json.dumps(
            {
                "notes": [
                    {"start": 1.0, "pitch": 45, "string": 1, "fret": 0},  # right
                    {"start": 1.5, "pitch": 69, "string": 5, "fret": 5},  # wrong
                    {"start": 10.0, "pitch": 69, "string": 5, "fret": 5},  # unverified passage
                ]
            }
        )
    )
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "aaa11111111.json").write_text(
        json.dumps(
            {
                "passages": [
                    {
                        "label": "stated",
                        "start": 0.0,
                        "end": 5.0,
                        "vocabulary": ["A0"],
                        "confidence": "stated",
                    },
                    {
                        "label": "guess",
                        "start": 9.0,
                        "end": 11.0,
                        "vocabulary": ["A0"],
                        "confidence": "unverified",
                    },
                ]
            }
        )
    )
    report = evaluate_video(paths, eval_dir)
    labels = [p["label"] for p in report["passages"]]
    assert labels == ["stated", "guess"]
    assert report["summary"]["passages"] == 1
    assert report["summary"]["passages_unverified"] == 1
    assert report["summary"]["notes"] == 2
    assert report["summary"]["position_precision"] == 0.5
