"""ClipCraft CLI — AI-powered video clipping from the terminal."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .pipeline import run_pipeline
from .transcribe import transcribe, save_transcript, load_transcript
from .analyze import find_clips, save_clips, load_clips
from .extract import extract_clip, get_video_info
from .caption import burn_captions, STYLES

console = Console()


def _print_banner():
    """Print ClipCraft banner."""
    banner = Text()
    banner.append("🎬 ", style="bold")
    banner.append("ClipCraft", style="bold cyan")
    banner.append(f" v{__version__}", style="dim")
    banner.append(" — AI video clipping for creators", style="dim")
    console.print(banner)
    console.print()


@click.group()
@click.version_option(__version__, "-V", "--version")
def main():
    """ClipCraft — AI-powered video clipping CLI.

    Transcribe, find viral moments, extract clips, burn captions.
    Built for creators who want open-source control over their short-form pipeline.

    \b
    Quick start:
      clipcraft full video.mp4                    # End-to-end pipeline
      clipcraft transcribe video.mp4               # Just transcribe
      clipcraft find video.mp4 --transcript t.json # Just find clips
      clipcraft extract video.mp4 --clip 1         # Extract one clip
    """
    pass


@main.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("-m", "--model", default="base",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              help="Whisper model size (default: base)")
@click.option("-l", "--language", default=None, help="Language code (auto-detect if not set)")
@click.option("-o", "--output", default=None, help="Output JSON path (default: <video>_transcript.json)")
@click.option("--api", is_flag=True, help="Use OpenAI Whisper API instead of local")
@click.option("--device", default="cpu", help="Device for local model (cpu/cuda)")
def transcribe_cmd(video, model, language, output, api, device):
    """Transcribe a video to text with word-level timestamps.

    Uses faster-whisper locally (free) by default.
    Pass --api to use OpenAI Whisper API (needs OPENAI_API_KEY).
    """
    video_path = Path(video)
    if output is None:
        output = f"{video_path.stem}_transcript.json"

    _print_banner()
    console.print(f"🎙️  Transcribing: [bold]{video}[/bold]")
    console.print(f"   Model: whisper-{model}, Device: {device}")
    if api:
        console.print("   Using OpenAI API (paid)")
    console.print()

    with console.status("[bold green]Transcribing...[/bold green]"):
        t = transcribe(
            video_path, model_size=model, language=language,
            device=device, use_api=api,
        )
        save_transcript(t, output)

    console.print()
    console.print(f"✅ Transcription complete!")
    console.print(f"   Language: {t.language}")
    console.print(f"   Duration: {t.duration:.0f}s")
    console.print(f"   Segments: {len(t.segments)}")
    console.print(f"   Saved to: [bold]{output}[/bold]")
    console.print(f"   Text preview: {t.text[:200]}...")


@main.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("-t", "--transcript", "transcript_path", default=None,
              help="Path to transcript JSON (auto-generates if not provided)")
@click.option("-n", "--num-clips", default=5, show_default=True,
              help="Number of clips to find")
@click.option("--min-len", default=30, show_default=True,
              help="Minimum clip length in seconds")
@click.option("--max-len", default=90, show_default=True,
              help="Maximum clip length in seconds")
@click.option("-o", "--output", default=None, help="Output JSON path")
@click.option("--whisper-model", default="base",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              help="Whisper model if transcribing")
def find_cmd(video, transcript_path, num_clips, min_len, max_len, output, whisper_model):
    """Find the most viral-worthy clips in a video.

    Uses an LLM to analyze the transcript and identify moments that would
    perform well as short-form content.
    """
    video_path = Path(video)

    if transcript_path is None:
        transcript_path = f"{video_path.stem}_transcript.json"

    if output is None:
        output = f"{video_path.stem}_clips.json"

    _print_banner()

    # Load or generate transcript
    tp = Path(transcript_path)
    if tp.exists():
        console.print(f"📝 Using transcript: [bold]{transcript_path}[/bold]")
        transcript = load_transcript(tp)
    else:
        console.print(f"🎙️  Transcribing first...")
        transcript = transcribe(video_path, model_size=whisper_model)
        save_transcript(transcript, tp)
        console.print(f"   ✓ Saved to {transcript_path}")

    console.print(f"🤖 Finding top {num_clips} clips ({min_len}s-{max_len}s)...")
    console.print()

    with console.status("[bold green]Analyzing with LLM...[/bold green]"):
        clips = find_clips(transcript, num_clips=num_clips, min_clip_len=min_len, max_clip_len=max_len)
        save_clips(clips, output)

    # Display results
    table = Table(title=f"🎬 Top Clips from {video_path.name}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="bold cyan", width=5)
    table.add_column("Title", style="bold")
    table.add_column("Time", width=12)
    table.add_column("Why", style="dim")

    for i, clip in enumerate(clips, 1):
        score_color = "green" if clip.score >= 8 else "yellow" if clip.score >= 6 else "red"
        table.add_row(
            str(i),
            f"[{score_color}]{clip.score}/10[/{score_color}]",
            clip.title,
            f"{clip.start:.0f}s→{clip.end:.0f}s",
            clip.why[:100],
        )

    console.print(table)
    console.print(f"\n✅ Saved to [bold]{output}[/bold]")


@main.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("-c", "--clips", "clips_path", required=True,
              help="Path to clips JSON (from 'find' command)")
@click.option("-t", "--transcript", "transcript_path", required=True,
              help="Path to transcript JSON")
@click.option("--clip", type=int, default=None,
              help="Extract a specific clip by number (1-based). Omit to extract all.")
@click.option("-s", "--style", default="default",
              type=click.Choice(list(STYLES.keys())),
              help="Caption style")
@click.option("-o", "--output-dir", default="clips", show_default=True,
              help="Output directory for clips")
def extract_cmd(video, clips_path, transcript_path, clip, style, output_dir):
    """Extract clips with burned captions.

    Takes a clips JSON (from 'find') and transcript, outputs finished clips.
    """
    _print_banner()

    clips = load_clips(clips_path)
    transcript = load_transcript(transcript_path)

    if clip is not None:
        if clip < 1 or clip > len(clips):
            console.print(f"[red]Error:[/red] Clip {clip} out of range (1-{len(clips)})")
            sys.exit(1)
        clips = [clips[clip - 1]]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(video)
    stem = video_path.stem

    for i, c in enumerate(clips):
        idx = clip or (i + 1)
        output = output_dir / f"{stem}_clip{idx:02d}.mp4"

        console.print(f"🎬 Clip {idx}: [bold]{c.title}[/bold] ({c.start:.0f}s→{c.end:.0f}s)")

        if output.exists():
            console.print(f"   ⏭️  Already exists, skipping")
            continue

        with console.status(f"[bold green]Extracting...[/bold green]"):
            extract_clip(video_path, output, c.start, c.end)
        console.print("   ✓ Extracted")

        with console.status(f"[bold green]Burning captions ({style})...[/bold green]"):
            burn_captions(video_path, output, transcript, c.start, c.end, style=style)
        console.print(f"   ✓ Captions burned")

    console.print(f"\n✅ Clips in [bold]{output_dir}/[/bold]")


@main.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("-n", "--num-clips", default=5, show_default=True,
              help="Number of clips to generate")
@click.option("--min-len", default=30, show_default=True,
              help="Minimum clip length in seconds")
@click.option("--max-len", default=90, show_default=True,
              help="Maximum clip length in seconds")
@click.option("-s", "--style", default="default",
              type=click.Choice(list(STYLES.keys())),
              help="Caption style")
@click.option("-o", "--output-dir", default="clips", show_default=True,
              help="Output directory")
@click.option("-m", "--whisper-model", default="base",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              help="Whisper model size")
@click.option("--device", default="cpu", help="Device for local model")
@click.option("--transcribe-only", is_flag=True,
              help="Stop after transcription")
@click.option("--analyze-only", is_flag=True,
              help="Stop after analysis (no extraction)")
@click.option("--no-resume", is_flag=True,
              help="Don't use cached transcript/clips")
@click.option("--start-from", type=int, default=None,
              help="Skip to step (1=transcribe, 2=analyze, 3=extract)")
def full(video, num_clips, min_len, max_len, style, output_dir, whisper_model,
         device, transcribe_only, analyze_only, no_resume, start_from):
    """Run the full pipeline: transcribe → analyze → extract → caption.

    This is the one-command experience. Give it a long video, get back
    short-form clips with burned captions.

    \b
    Examples:
      clipcraft full podcast.mp4
      clipcraft full talk.mp4 -n 3 --style tiktok
      clipcraft full video.mp4 --transcribe-only  # Just get the transcript
    """
    _print_banner()
    console.print(f"📹 Input: [bold]{video}[/bold]")
    console.print(f"🎯 Goal: {num_clips} clips ({min_len}s-{max_len}s)")
    console.print(f"🎨 Style: {style}")
    console.print(f"📂 Output: {output_dir}/")
    console.print()

    results = run_pipeline(
        video,
        output_dir=output_dir,
        num_clips=num_clips,
        min_clip_len=min_len,
        max_clip_len=max_len,
        caption_style=style,
        whisper_model=whisper_model,
        device=device,
        transcribe_only=transcribe_only,
        analyze_only=analyze_only,
        resume=not no_resume,
        start_from=start_from,
    )

    if transcribe_only or analyze_only:
        console.print(f"\n✅ Done! Results in [bold]{output_dir}/[/bold]")
        return

    # Summary table
    table = Table(title="✨ Generated Clips")
    table.add_column("File", style="bold")
    table.add_column("Title")
    table.add_column("Duration", justify="right")
    table.add_column("Score", justify="center")

    for r in results:
        if r.get("skipped"):
            continue
        filename = Path(r["path"]).name
        score_str = f"[green]{r['score']}/10[/green]" if r["score"] >= 7 else f"[yellow]{r['score']}/10[/yellow]"
        table.add_row(filename, r["title"], f"{r['duration']:.0f}s", score_str)

    console.print()
    console.print(table)
    console.print(f"\n✅ [bold]{len(results)} clips[/bold] ready in [bold]{output_dir}/[/bold]")
    console.print("   Drop them into TikTok, Reels, Shorts, or your editing tool of choice.")


@main.command()
@click.argument("video", type=click.Path(exists=True))
def info(video):
    """Show video metadata (resolution, duration, codec)."""
    _print_banner()
    info_data = get_video_info(video)

    console.print(f"📹 [bold]{video}[/bold]")
    console.print()

    fmt = info_data.get("format", {})
    console.print(f"  Duration: {float(fmt.get('duration', 0)):.1f}s")
    console.print(f"  Size: {int(fmt.get('size', 0)) / (1024*1024):.0f} MB")
    console.print(f"  Format: {fmt.get('format_name', 'unknown')}")
    console.print()

    for stream in info_data.get("streams", []):
        if stream["codec_type"] == "video":
            console.print(f"  📺 Video: {stream.get('width')}×{stream.get('height')}, "
                         f"{stream.get('codec_name')}, "
                         f"{stream.get('r_frame_rate', '').split('/')[0]}fps")
        elif stream["codec_type"] == "audio":
            console.print(f"  🎵 Audio: {stream.get('codec_name')}, "
                         f"{stream.get('sample_rate')}Hz, "
                         f"{stream.get('channels', 0)}ch")


@main.command()
def styles():
    """List available caption styles."""
    _print_banner()
    console.print("🎨 Available caption styles:\n")

    for name, style in STYLES.items():
        console.print(f"  [bold]{name}[/bold]")
        console.print(f"    Font: {style.font_name} {style.font_size}px")
        console.print(f"    Color: {style.font_color} / highlight: {style.highlight_color}")
        console.print(f"    Position: {style.position}, background: {style.background}")
        console.print()


if __name__ == "__main__":
    main()
