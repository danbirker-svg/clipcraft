"""Clip extraction — FFmpeg-based video extraction with trimming and formatting."""

import subprocess
from pathlib import Path


def extract_clip(
    input_video: str | Path,
    output_video: str | Path,
    start: float,
    end: float,
    video_width: int = 1080,
    video_height: int = 1920,
    fps: int = 30,
    crf: int = 23,
    preset: str = "fast",
) -> str:
    """Extract a clip from a video, reformat to vertical (9:16) short-form.

    Args:
        input_video: Source video file.
        output_video: Output file path.
        start: Clip start time in seconds.
        end: Clip end time in seconds.
        video_width: Output width (default 1080 for vertical).
        video_height: Output height (default 1920 for vertical).
        fps: Output frame rate.
        crf: Quality (lower = better, 18-28 is good range).
        preset: FFmpeg preset (fast/medium/slow).

    Returns:
        Path to the output file.
    """
    duration = end - start

    if duration <= 0:
        raise ValueError(f"Invalid clip duration: {end} - {start} = {duration}s")

    # Auto-crop to vertical: center crop of the source
    # For 16:9 source → 9:16 vertical: crop the center
    scale_filter = (
        f"scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
        f"crop={video_width}:{video_height}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0, start - 0.5)),  # Small preroll for seeking accuracy
        "-i", str(input_video),
        "-ss", "0.5",                       # Compensate for preroll
        "-t", str(duration),
        "-vf", scale_filter,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        "-r", str(fps),
        str(output_video),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(output_video)


def extract_audio(
    input_video: str | Path,
    output_audio: str | Path,
    start: float | None = None,
    end: float | None = None,
) -> str:
    """Extract audio from a video, optionally trimmed.

    Useful for faster transcription — whisper only needs audio.
    """
    cmd = ["ffmpeg", "-y", "-i", str(input_video)]

    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        cmd.extend(["-t", str(end - start if start else end)])

    cmd.extend([
        "-vn",                    # No video
        "-acodec", "pcm_s16le",  # WAV for best whisper compatibility
        "-ar", "16000",           # 16kHz mono for whisper
        "-ac", "1",
        str(output_audio),
    ])

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(output_audio)


def get_video_info(path: str | Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    import json
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)
