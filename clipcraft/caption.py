"""Caption burning — renders word-level captions onto video clips.

Styles: default (clean white text with highlight), tiktok (bold with background),
minimal (small bottom text), karaoke (word-by-word color sweep).
"""

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .transcribe import Transcript, Word, Segment


@dataclass
class CaptionStyle:
    """Caption rendering style."""
    name: str
    font_size: int = 48
    font_color: str = "white"
    highlight_color: str = "#FFD700"  # gold
    background: str = "none"           # none, box, full_width
    position: str = "center"           # center, top, bottom
    max_chars_per_line: int = 30
    font_name: str = "Arial"


STYLES = {
    "default": CaptionStyle(
        name="default",
        font_size=48,
        font_color="white",
        highlight_color="#FFD700",
        background="none",
        position="center",
    ),
    "tiktok": CaptionStyle(
        name="tiktok",
        font_size=52,
        font_color="white",
        highlight_color="#FF0050",
        background="box",
        position="center",
    ),
    "minimal": CaptionStyle(
        name="minimal",
        font_size=28,
        font_color="white",
        background="box",
        position="bottom",
    ),
    "karaoke": CaptionStyle(
        name="karaoke",
        font_size=44,
        font_color="#888888",
        highlight_color="#00FF88",
        background="none",
        position="center",
    ),
}


def _words_for_timerange(
    segments: list[Segment],
    start_time: float,
    end_time: float,
    pad_seconds: float = 0.5,
) -> list[Word]:
    """Get words within a time range, with padding."""
    words = []
    window_start = max(0, start_time - pad_seconds)

    for seg in segments:
        if seg.end < window_start:
            continue
        if seg.start > end_time:
            break

        for w in seg.words:
            if window_start <= w.start <= end_time:
                words.append(w)

    return words


def _words_to_lines(words: list[Word], max_chars: int = 30) -> list[list[Word]]:
    """Group words into display lines based on character count."""
    lines = []
    current_line = []
    current_chars = 0

    for w in words:
        word_len = len(w.word)
        if current_chars + word_len + (1 if current_line else 0) > max_chars:
            if current_line:
                lines.append(current_line)
            current_line = [w]
            current_chars = word_len
        else:
            current_line.append(w)
            current_chars += word_len + (1 if current_line else 0)

    if current_line:
        lines.append(current_line)

    return lines


def _build_ass(
    words: list[Word],
    clip_start: float,
    style: CaptionStyle,
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """Build an ASS subtitle string with word-level karaoke highlighting.

    Uses ASS (Advanced SubStation Alpha) format with \\k tags for word-by-word
    timing. FFmpeg can burn these in with the subtitles filter.
    """
    lines = _words_to_lines(words, style.max_chars_per_line)

    # Calculate vertical position
    if style.position == "center":
        # Center of bottom third (typical TikTok placement)
        base_y = int(video_height * 0.65)
        # Shift up based on number of lines
        line_height = style.font_size + 8
        base_y -= (len(lines) - 1) * line_height // 2
    elif style.position == "top":
        base_y = int(video_height * 0.15)
    else:  # bottom
        base_y = int(video_height * 0.85)

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},&H00FFFFFF,&H00000000,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,2,50,50,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    clip_offset_ms = clip_start * 1000
    line_height = style.font_size + 10

    for line_idx, line_words in enumerate(lines):
        if not line_words:
            continue

        line_y = base_y + (line_idx * line_height)

        # Build karaoke timing for each word
        ass_text_parts = []
        for w in line_words:
            word_start_ms = int((w.start - clip_start) * 1000)
            word_duration_cs = max(1, int((w.end - w.start) * 100))  # centiseconds
            word_text = w.word.replace("{", "\\{").replace("}", "\\}")
            ass_text_parts.append(f"{{\\k{word_duration_cs}}}{word_text}")

        line_text = " ".join(ass_text_parts)

        # Build background box if needed
        bg_tag = ""
        if style.background in ("box", "full_width"):
            # Approximate background with border + shadow (ASS doesn't do per-word bg natively)
            bg_tag = "{\\bord2\\shad0\\3c&H80000000}"

        # Build highlight color override
        hl_tag = f"{{\\1c&H{_rgb_to_bgr(style.highlight_color)}&}}"

        line_start_ms = int((line_words[0].start - clip_start) * 1000)
        line_end_ms = int((line_words[-1].end - clip_start) * 1000)

        # Font color
        color_tag = ""
        if style.font_color != "white":
            color_tag = f"{{\\1c&H{_rgb_to_bgr(style.font_color)}&}}"

        # ASS event with position override
        pos_tag = f"{{\\an2\\pos({video_width // 2},{line_y})}}"

        events.append(
            f"Dialogue: 0,{_ms_to_ass(line_start_ms)},{_ms_to_ass(line_end_ms)},"
            f"Default,,0,0,0,,{pos_tag}{bg_tag}{color_tag}{hl_tag}{line_text}"
        )

    return ass_header + "\n".join(events)


def _rgb_to_bgr(hex_color: str) -> str:
    """Convert #RRGGBB to BGR hex for ASS format."""
    hex_color = hex_color.lstrip("#")
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"{b}{g}{r}"


def _ms_to_ass(ms: int) -> str:
    """Convert milliseconds to ASS timestamp (H:MM:SS.cc)."""
    ms = max(0, ms)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    cs = (ms % 1000) // 10
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def burn_captions(
    input_video: str | Path,
    output_video: str | Path,
    transcript: Transcript,
    clip_start: float,
    clip_end: float,
    style: str | CaptionStyle = "default",
    video_width: int = 1080,
    video_height: int = 1920,
    pre_trimmed: bool = False,
) -> str:
    """Burn word-level captions onto a video clip.

    Args:
        input_video: Source video file.
        output_video: Output file path.
        transcript: Full transcript with word timestamps.
        clip_start: Start of the clip in seconds (in the ORIGINAL video's timeline —
            used to look up transcript words and offset caption timing).
        clip_end: End of the clip in seconds (original timeline).
        style: Caption style name or CaptionStyle object.
        video_width: Output video width.
        video_height: Output video height.
        pre_trimmed: Set True if input_video is ALREADY trimmed to [clip_start, clip_end]
            (i.e. it starts at 0). Skips ffmpeg seeking — without this, seeking
            clip_start seconds into a short pre-trimmed clip produces an empty MP4.

    Returns:
        Path to the output video.
    """
    import subprocess

    if isinstance(style, str):
        style = STYLES.get(style, STYLES["default"])

    # Seek args: only seek into the input if it's the full-length source video.
    seek_args = [] if pre_trimmed else ["-ss", str(clip_start)]
    duration_args = [] if pre_trimmed else ["-t", str(clip_end - clip_start)]

    # Get words for this clip's time range
    words = _words_for_timerange(transcript.segments, clip_start, clip_end)

    if not words:
        # No captions to burn — just copy the clip
        subprocess.run([
            "ffmpeg", "-y",
            *seek_args,
            "-i", str(input_video),
            *duration_args,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            "-vf", f"scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2",
            str(output_video),
        ], check=True, capture_output=True)
        return str(output_video)

    # Build ASS subtitle file
    ass_content = _build_ass(words, clip_start, style, video_width, video_height)

    with tempfile.NamedTemporaryFile(suffix=".ass", mode="w", delete=False) as f:
        f.write(ass_content)
        ass_path = f.name

    try:
        # FFmpeg: extract clip + burn subtitles + scale
        cmd = [
            "ffmpeg", "-y",
            *seek_args,
            "-i", str(input_video),
            *duration_args,
            "-vf",
            f"scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
            f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2,"
            f"ass={ass_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(output_video),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        os.unlink(ass_path)

    return str(output_video)


def burn_captions_simple(
    input_video: str | Path,
    output_video: str | Path,
    captions: list[dict],  # [{text: str, start: float, end: float}]
    style: str = "default",
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """Burn captions without a full transcript — simpler API for pre-processed captions.

    Each caption dict: {"text": "Hello world", "start": 1.5, "end": 3.2}
    """
    import subprocess
    import tempfile

    if isinstance(style, str):
        style = STYLES.get(style, STYLES["default"])

    # Build a simplified transcript from the caption list
    segments = []
    for cap in captions:
        text = cap["text"]
        start = cap["start"]
        end = cap["end"]
        # Fake word-level timing: split text into words, distribute evenly
        words_list = text.split()
        if words_list:
            word_dur = (end - start) / len(words_list)
            words = [
                Word(word=w, start=start + i * word_dur, end=start + (i + 1) * word_dur, confidence=1.0)
                for i, w in enumerate(words_list)
            ]
        else:
            words = []
        segments.append(Segment(text=text, start=start, end=end, words=words))

    t = Transcript(text=" ".join(c["text"] for c in captions), segments=segments, language="en", duration=0)

    return burn_captions(
        input_video, output_video, t,
        clip_start=min(c["start"] for c in captions),
        clip_end=max(c["end"] for c in captions),
        style=style,
        video_width=video_width,
        video_height=video_height,
    )
