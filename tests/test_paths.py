"""Tests for paths.py — video-id extraction, slug generation, VideoPaths."""

from __future__ import annotations

import json

import pytest

from migs_tab.paths import VideoPaths, extract_video_id, slugify_title


class TestExtractVideoId:
    def test_bare_id(self):
        assert extract_video_id("wS_i91qxQYM") == "wS_i91qxQYM"

    def test_full_watch_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=wS_i91qxQYM") == "wS_i91qxQYM"

    def test_watch_url_with_extra_params(self):
        assert (
            extract_video_id("https://www.youtube.com/watch?v=wS_i91qxQYM&t=42s&feature=shared")
            == "wS_i91qxQYM"
        )

    def test_short_youtu_be(self):
        assert extract_video_id("https://youtu.be/wS_i91qxQYM") == "wS_i91qxQYM"

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/wS_i91qxQYM") == "wS_i91qxQYM"

    def test_shorts_url(self):
        assert extract_video_id("https://www.youtube.com/shorts/wS_i91qxQYM") == "wS_i91qxQYM"

    def test_nocookie_domain(self):
        assert (
            extract_video_id("https://www.youtube-nocookie.com/watch?v=wS_i91qxQYM")
            == "wS_i91qxQYM"
        )

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("https://example.com/foo")

    def test_invalid_short_id_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("notavideoidat")  # wrong length

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("aaaa$aaaaaaa")  # invalid char


class TestSlugifyTitle:
    def test_basic(self):
        assert slugify_title("Hello World") == "hello-world"

    def test_punctuation_collapsed(self):
        assert slugify_title("Hello!! World??") == "hello-world"

    def test_repeated_dashes_collapsed(self):
        assert slugify_title("foo---bar___baz") == "foo-bar-baz"

    def test_strip_edges(self):
        assert slugify_title("---Hello---") == "hello"

    def test_unicode_normalized(self):
        assert slugify_title("Café Crème") == "cafe-creme"

    def test_truncated(self):
        long = "a" * 200
        result = slugify_title(long, max_length=50)
        assert len(result) <= 50

    def test_truncated_strips_trailing_dash(self):
        result = slugify_title("hello world " * 20, max_length=11)
        assert not result.endswith("-")


class TestVideoPaths:
    def test_creates_cache_dir(self, tmp_path):
        paths = VideoPaths("abc12345678", cache_dir=tmp_path / "cache")
        assert paths.root.exists()
        assert paths.root.name == "abc12345678"

    def test_default_paths(self, tmp_path):
        paths = VideoPaths("abc12345678", cache_dir=tmp_path / "cache")
        assert paths.video.name == "video.mp4"
        assert paths.audio.name == "audio.wav"
        assert paths.captions_vtt.name == "captions.en.vtt"
        assert paths.notes_midi.name == "notes.mid"
        assert paths.notes_json.name == "notes.json"
        assert paths.structure_json.name == "structure.json"
        assert paths.tuning_json.name == "tuning.json"
        assert paths.frets_json.name == "frets.json"
        assert paths.tips_md.name == "tips.md"

    def test_output_dir_uses_title_slug(self, tmp_path):
        paths = VideoPaths("abc12345678", cache_dir=tmp_path / "cache")
        # Write a minimal info.json so the title-slug path activates.
        paths.info_json.write_text(json.dumps({"title": "Hello World Tutorial"}))
        out_root = tmp_path / "output"
        d = paths.output_dir(out_root)
        # Suffixed with video_id so two titles with the same name don't collide.
        assert d.name == "hello-world-tutorial-abc12345678"
        assert d.exists()

    def test_output_dir_falls_back_to_id(self, tmp_path):
        paths = VideoPaths("abc12345678", cache_dir=tmp_path / "cache")
        d = paths.output_dir(tmp_path / "output")
        assert d.name == "abc12345678"
