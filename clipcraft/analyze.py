"""Clip analysis — uses an LLM to identify viral-worthy moments from transcripts.

Feeds the transcript with timestamps to an LLM and asks it to find the most
clip-worthy segments. Returns structured results with start/end times, titles,
and reasons.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from .transcribe import Transcript


@dataclass
class ClipCandidate:
    """A clip-worthy moment identified by the LLM."""
    start: float       # seconds
    end: float         # seconds
    title: str         # suggested clip title
    hook: str          # the hook line (first 1-2 sentences)
    why: str           # why this moment would go viral
    category: str      # surprise, emotion, tip, controversy, humor, story
    score: int         # 1-10 viral potential


ANALYSIS_PROMPT = """You are a professional short-form video editor. Analyze the transcript below 
and identify the {num_clips} most clip-worthy moments for short-form content (TikTok, Reels, Shorts).

For each moment, look for:
- **Strong hooks**: first 5 seconds that grab attention
- **Surprising insights**: counter-intuitive facts or revelations
- **Emotional peaks**: anger, excitement, laughter, vulnerability
- **Actionable advice**: tactical tips the viewer can use immediately
- **Polarizing takes**: opinions that spark debate
- **Great stories**: compelling narratives with setup → conflict → resolution

The transcript has word-level timestamps. Output exact start/end times.

Transcript duration: {duration:.0f}s
Target clip length: {min_clip_len}s to {max_clip_len}s

Output as JSON array:
[
  {{
    "start": 123.4,
    "end": 183.4,
    "title": "Short, punchy title",
    "hook": "The exact hook line from the transcript",
    "why": "Why this would perform well on short-form platforms",
    "category": "surprise|emotion|tip|controversy|humor|story",
    "score": 8
  }}
]

TRANSCRIPT:
{transcript}"""


def _build_transcript_for_llm(transcript: Transcript, max_chars: int = 15_000) -> str:
    """Build a timestamped transcript string for the LLM prompt."""
    lines = []
    chars = 0

    for seg in transcript.segments:
        if seg.words:
            # Word-level timestamps
            line_parts = []
            for w in seg.words:
                line_parts.append(f"{w.word}")
            line = f"[{seg.start:.1f}s] {' '.join(line_parts)}"
        else:
            line = f"[{seg.start:.1f}s] {seg.text}"

        if chars + len(line) > max_chars:
            break

        lines.append(line)
        chars += len(line)

    return "\n".join(lines)


def _call_llm(prompt: str, provider: str = "openrouter") -> list[dict]:
    """Call an LLM to analyze the transcript.

    Uses OpenRouter by default (set OPENROUTER_API_KEY).
    Falls back to OpenAI (set OPENAI_API_KEY).
    """
    import httpx

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY to use LLM analysis."
        )

    if os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter (cheaper, more model choice)
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("CLIPCRAFT_MODEL", "anthropic/claude-sonnet-4"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
    else:
        # Fallback to OpenAI
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )

    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Parse JSON from response (may be wrapped in ```json blocks)
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    return json.loads(content)


def find_clips(
    transcript: Transcript,
    num_clips: int = 5,
    min_clip_len: int = 30,
    max_clip_len: int = 90,
    provider: str = "openrouter",
) -> list[ClipCandidate]:
    """Analyze transcript and find the best clip-worthy moments.

    Args:
        transcript: The transcribed video.
        num_clips: Number of clips to find (default 5).
        min_clip_len: Minimum clip length in seconds (default 30).
        max_clip_len: Maximum clip length in seconds (default 90).
        provider: LLM provider ('openrouter' or 'openai').

    Returns:
        List of ClipCandidate objects sorted by viral score.
    """
    transcript_str = _build_transcript_for_llm(transcript)
    prompt = ANALYSIS_PROMPT.format(
        num_clips=num_clips,
        duration=transcript.duration,
        min_clip_len=min_clip_len,
        max_clip_len=max_clip_len,
        transcript=transcript_str,
    )

    raw = _call_llm(prompt, provider=provider)

    # Handle different response shapes
    if isinstance(raw, dict):
        items = raw.get("clips") or raw.get("moments") or [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unexpected LLM response type: {type(raw)}")

    candidates = []
    for item in items:
        try:
            candidates.append(ClipCandidate(
                start=float(item["start"]),
                end=float(item["end"]),
                title=str(item.get("title", "Untitled Clip")),
                hook=str(item.get("hook", "")),
                why=str(item.get("why", "")),
                category=str(item.get("category", "tip")),
                score=int(item.get("score", 5)),
            ))
        except (KeyError, ValueError, TypeError) as e:
            # Skip malformed entries
            continue

    # Sort by viral score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def save_clips(clips: list[ClipCandidate], path: str) -> None:
    """Save clip candidates to JSON file."""
    data = [
        {
            "start": c.start,
            "end": c.end,
            "title": c.title,
            "hook": c.hook,
            "why": c.why,
            "category": c.category,
            "score": c.score,
        }
        for c in clips
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_clips(path: str) -> list[ClipCandidate]:
    """Load clip candidates from JSON file."""
    with open(path) as f:
        data = json.load(f)

    return [
        ClipCandidate(
            start=c["start"],
            end=c["end"],
            title=c["title"],
            hook=c["hook"],
            why=c["why"],
            category=c["category"],
            score=c["score"],
        )
        for c in data
    ]
