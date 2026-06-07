"""End-to-end pipeline — orchestrate transcribe → analyze → extract → caption."""

import json
import os
from pathlib import Path
from typing import Optional

from .transcribe import transcribe, load_transcript, save_transcript, Transcript
from .analyze import find_clips, save_clips, load_clips, ClipCandidate
from .extract import extract_clip
from .caption import burn_captions, CaptionStyle, STYLES


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path = "clips",
    num_clips: int = 5,
    min_clip_len: int = 30,
    max_clip_len: int = 90,
    caption_style: str = "default",
    whisper_model: str = "base",
    transcribe_only: bool = False,
    analyze_only: bool = False,
    resume: bool = True,
    device: str = "cpu",
    start_from: Optional[int] = None,
) -> list[dict]:
    """Run the full clipcraft pipeline.

    Args:
        video_path: Source video file.
        output_dir: Directory for outputs.
        num_clips: Number of clips to generate.
        min_clip_len: Minimum clip duration in seconds.
        max_clip_len: Maximum clip duration in seconds.
        caption_style: Caption style name ('default', 'tiktok', 'minimal', 'karaoke').
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        transcribe_only: Stop after transcription.
        analyze_only: Skip clip extraction (transcribe + analyze only).
        resume: Load cached transcript/clips if available.
        device: 'cpu' or 'cuda' for whisper.
        start_from: Skip to a specific step (1=transcribe, 2=analyze, 3=extract).

    Returns:
        List of generated clip info dicts.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    transcript_path = output_dir / f"{stem}_transcript.json"
    clips_path = output_dir / f"{stem}_clips.json"

    transcript = None
    clips = None

    # Step 1: Transcribe
    if start_from is None or start_from <= 1:
        if resume and transcript_path.exists():
            print(f"📝 Loading cached transcript: {transcript_path}")
            transcript = load_transcript(transcript_path)
        else:
            print(f"🎙️  Transcribing {video_path} with whisper-{whisper_model}...")
            transcript = transcribe(
                video_path,
                model_size=whisper_model,
                device=device,
            )
            save_transcript(transcript, transcript_path)
            print(f"   ✓ Saved transcript ({len(transcript.segments)} segments, {transcript.duration:.0f}s)")
    else:
        transcript = load_transcript(transcript_path)

    if transcribe_only:
        return [{"transcript_path": str(transcript_path)}]

    # Step 2: Analyze
    if start_from is None or start_from <= 2:
        if resume and clips_path.exists():
            print(f"🔍 Loading cached clip analysis: {clips_path}")
            clips = load_clips(clips_path)
        else:
            print(f"🤖 Analyzing transcript for top {num_clips} clip-worthy moments...")
            clips = find_clips(
                transcript,
                num_clips=num_clips,
                min_clip_len=min_clip_len,
                max_clip_len=max_clip_len,
            )
            save_clips(clips, clips_path)
        print(f"   ✓ Found {len(clips)} clips")
    else:
        clips = load_clips(clips_path)

    if analyze_only:
        return [
            {
                "transcript_path": str(transcript_path),
                "clips_path": str(clips_path),
                "clips": [
                    {"title": c.title, "start": c.start, "end": c.end, "score": c.score, "why": c.why}
                    for c in clips
                ],
            }
        ]

    # Step 3: Extract + Caption
    results = []
    for i, clip in enumerate(clips, 1):
        clip_output = output_dir / f"{stem}_clip{i:02d}_{_slugify(clip.title)}.mp4"
        print(f"\n🎬 Clip {i}/{len(clips)}: {clip.title}")
        print(f"   ⏱️  {clip.start:.1f}s → {clip.end:.1f}s ({clip.end - clip.start:.0f}s)")

        if clip_output.exists():
            print(f"   ⏭️  Already exists, skipping")
            results.append({
                "title": clip.title,
                "path": str(clip_output),
                "start": clip.start,
                "end": clip.end,
                "score": clip.score,
                "skipped": True,
            })
            continue

        # Extract the clip
        temp_clip = output_dir / f"{stem}_clip{i:02d}_raw.mp4"
        extract_clip(video_path, temp_clip, clip.start, clip.end)
        print(f"   ✓ Extracted")

        # Burn captions
        burn_captions(
            temp_clip, clip_output,
            transcript, clip.start, clip.end,
            style=caption_style,
        )
        print(f"   ✓ Captions burned ({caption_style} style)")

        # Clean up raw clip
        temp_clip.unlink(missing_ok=True)

        results.append({
            "title": clip.title,
            "path": str(clip_output),
            "start": clip.start,
            "end": clip.end,
            "duration": clip.end - clip.start,
            "score": clip.score,
            "category": clip.category,
            "hook": clip.hook,
            "why": clip.why,
        })

    print(f"\n✅ Done! {len(results)} clips in {output_dir}/")
    return results


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text[:50]
