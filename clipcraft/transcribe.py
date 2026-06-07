"""Transcription module — converts video/audio to text with word-level timestamps.

Uses faster-whisper (local, free) by default. Falls back to OpenAI Whisper API
when OPENAI_API_KEY is set and --api is passed.
"""

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """A single word with its timestamp."""
    word: str
    start: float  # seconds
    end: float    # seconds
    confidence: float


@dataclass
class Segment:
    """A transcribed segment with words."""
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """Full transcription result."""
    text: str
    segments: list[Segment]
    language: str
    duration: float  # seconds


def transcribe_local(
    video_path: str | Path,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> Transcript:
    """Transcribe using faster-whisper (local, free)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_raw, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
    )

    segments = []
    full_text_parts = []
    duration = info.duration if hasattr(info, "duration") else 0.0

    for seg in segments_raw:
        words = [
            Word(
                word=w.word.strip(),
                start=w.start,
                end=w.end,
                confidence=w.probability,
            )
            for w in (seg.words or [])
        ]
        text = seg.text.strip()
        full_text_parts.append(text)
        segments.append(Segment(
            text=text,
            start=seg.start,
            end=seg.end,
            words=words,
        ))

    return Transcript(
        text=" ".join(full_text_parts),
        segments=segments,
        language=info.language,
        duration=duration or segments[-1].end if segments else 0.0,
    )


def transcribe_api(
    video_path: str | Path,
    model: str = "whisper-1",
    language: Optional[str] = None,
) -> Transcript:
    """Transcribe using OpenAI Whisper API (paid, but word-level timestamps)."""
    from openai import OpenAI

    client = OpenAI()
    path = Path(video_path)

    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language=language,
        )

    segments = []
    full_text_parts = []

    for seg in result.segments:
        words = [
            Word(
                word=w.get("word", "").strip(),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
                confidence=w.get("confidence", 1.0),
            )
            for w in (seg.get("words") or [])
        ]
        text = seg.get("text", "").strip()
        full_text_parts.append(text)
        segments.append(Segment(
            text=text,
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            words=words,
        ))

    return Transcript(
        text=" ".join(full_text_parts),
        segments=segments,
        language=result.language,
        duration=segments[-1].end if segments else 0.0,
    )


def transcribe(
    video_path: str | Path,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "cpu",
    use_api: bool = False,
) -> Transcript:
    """Transcribe video/audio to text. Local by default, API with --api flag."""
    if use_api:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set it or remove --api to use local transcription."
            )
        return transcribe_api(video_path, language=language)

    return transcribe_local(
        video_path,
        model_size=model_size,
        language=language,
        device=device,
    )


def save_transcript(transcript: Transcript, path: str | Path) -> None:
    """Save transcript to JSON file for reuse."""
    data = {
        "text": transcript.text,
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in seg.words
                ],
            }
            for seg in transcript.segments
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_transcript(path: str | Path) -> Transcript:
    """Load transcript from JSON file."""
    with open(path) as f:
        data = json.load(f)

    segments = [
        Segment(
            text=seg["text"],
            start=seg["start"],
            end=seg["end"],
            words=[
                Word(w["word"], w["start"], w["end"], w.get("confidence", 0.0))
                for w in seg.get("words", [])
            ],
        )
        for seg in data["segments"]
    ]

    return Transcript(
        text=data["text"],
        segments=segments,
        language=data["language"],
        duration=data["duration"],
    )
