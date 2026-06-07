"""Tests for ClipCraft."""
import json
import tempfile
from pathlib import Path

import pytest

from clipcraft.transcribe import Transcript, Segment, Word, save_transcript, load_transcript
from clipcraft.analyze import ClipCandidate, save_clips, load_clips
from clipcraft.extract import get_video_info
from clipcraft.caption import STYLES, _words_for_timerange, _words_to_lines


class TestTranscriptIO:
    def test_save_and_load(self):
        t = Transcript(
            text="hello world",
            language="en",
            duration=10.0,
            segments=[
                Segment(
                    text="hello world",
                    start=0.0,
                    end=2.0,
                    words=[
                        Word("hello", 0.0, 0.5, 0.99),
                        Word("world", 0.6, 1.0, 0.98),
                    ],
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_transcript(t, path)
            loaded = load_transcript(path)

            assert loaded.text == "hello world"
            assert loaded.language == "en"
            assert loaded.duration == 10.0
            assert len(loaded.segments) == 1
            assert len(loaded.segments[0].words) == 2
            assert loaded.segments[0].words[0].word == "hello"
            assert loaded.segments[0].words[0].start == 0.0
        finally:
            Path(path).unlink()


class TestClipIO:
    def test_save_and_load(self):
        clips = [
            ClipCandidate(
                start=10.0, end=70.0,
                title="Amazing insight",
                hook="Here's the thing...",
                why="Surprising revelation",
                category="surprise",
                score=9,
            ),
            ClipCandidate(
                start=120.0, end=165.0,
                title="Hot take",
                hook="Everyone is wrong about...",
                why="Controversial opinion",
                category="controversy",
                score=7,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_clips(clips, path)
            loaded = load_clips(path)

            assert len(loaded) == 2
            assert loaded[0].title == "Amazing insight"
            assert loaded[0].score == 9
            assert loaded[1].category == "controversy"
        finally:
            Path(path).unlink()


class TestCaptionWords:
    def test_words_for_timerange(self):
        words = [
            Word("hello", 0.0, 0.5, 1.0),
            Word("world", 0.6, 1.0, 1.0),
            Word("this", 1.2, 1.5, 1.0),
            Word("is", 1.6, 1.8, 1.0),
            Word("clipcraft", 1.9, 2.5, 1.0),
        ]
        seg = Segment(text="hello world this is clipcraft", start=0.0, end=2.5, words=words)

        result = _words_for_timerange([seg], 0.3, 1.9)
        assert len(result) >= 2
        assert result[0].word == "hello"
        assert result[-1].word in ("is", "clipcraft")

    def test_words_to_lines(self):
        words = [Word("word", 0.0, 0.1, 1.0) for _ in range(20)]
        lines = _words_to_lines(words, max_chars=30)
        assert len(lines) > 1
        assert all(len(" ".join(w.word for w in line)) <= 30 for line in lines)


class TestStyles:
    def test_all_styles_present(self):
        assert "default" in STYLES
        assert "tiktok" in STYLES
        assert "minimal" in STYLES
        assert "karaoke" in STYLES

    def test_style_attributes(self):
        for name, style in STYLES.items():
            assert style.font_size > 0
            assert style.font_color
            assert style.position in ("center", "top", "bottom")


class TestPipeline:
    def test_slugify(self):
        from clipcraft.pipeline import _slugify
        assert _slugify("Hello World!") == "hello-world"
        assert _slugify("The 3 BEST Tips!!!") == "the-3-best-tips"
        assert _slugify("  Spaces  Everywhere  ") == "spaces-everywhere"
