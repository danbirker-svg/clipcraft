# ClipCraft 🎬

**AI-powered video clipping CLI — transcribe, find viral moments, extract clips, burn captions.**

Give ClipCraft a long video (podcast, talk, stream) and get back short-form clips ready for TikTok, Reels, and Shorts. Uses local whisper for **free transcription**, an LLM to identify the most viral-worthy moments, and FFmpeg to extract and caption your clips.

## Quick Start

```bash
# Install
pip install git+https://github.com/danbirker-svg/clipcraft.git

# One command: long video → short clips
clipcraft full podcast.mp4

# Or step by step
clipcraft transcribe podcast.mp4              # Whisper transcription (free, local)
clipcraft find podcast.mp4                    # LLM picks the best moments
clipcraft extract podcast.mp4 -c clips.json -t transcript.json  # Export with captions
```

Output:
```
clips/
├── podcast_clip01_the_contrarian_take.mp4    ← 60s clip, captions burned in
├── podcast_clip02_surprising_revelation.mp4  ← 45s clip
├── podcast_clip03_actionable_advice.mp4      ← 55s clip
├── podcast_clip04_funny_story.mp4            ← 35s clip
├── podcast_clip05_controversial_opinion.mp4  ← 50s clip
├── podcast_transcript.json                   ← Cached for reuse
└── podcast_clips.json                        ← Clip metadata
```

## How It Works

```
📹 Long Video (30+ min)
    │
    ▼
🎙️  Transcribe (faster-whisper, local & free)
    │  Word-level timestamps
    ▼
🤖 Analyze (LLM: Claude/GPT)
    │  Finds: hooks, surprises, emotions, tips, stories
    │  Scores each moment 1-10
    ▼
✂️  Extract (FFmpeg)
    │  Trims + reframes to 9:16 vertical
    ▼
💬 Caption (ASS subtitle burn)
    │  Word-by-word karaoke highlighting
    ▼
✨ 5 short-form clips, ready to post
```

## Installation

### Prerequisites

- **Python 3.10+**
- **FFmpeg** (`apt install ffmpeg` / `brew install ffmpeg`)
- **OpenRouter API key** (or OpenAI) for LLM clip analysis

```bash
# From PyPI
pip install git+https://github.com/danbirker-svg/clipcraft.git

# From source
git clone https://github.com/danbirker-svg/clipcraft.git
cd clipcraft
pip install -e .
```

### Whisper Model

On first run, faster-whisper downloads the model (~150MB for `base`). Faster models:

```bash
clipcraft full video.mp4 -m tiny    # ~75MB, fast, less accurate
clipcraft full video.mp4 -m base    # ~150MB, good balance (default)
clipcraft full video.mp4 -m small   # ~500MB, more accurate
clipcraft full video.mp4 -m medium  # ~1.5GB, very accurate (slower)
```

## Commands

### `clipcraft full` — End-to-End Pipeline

```bash
clipcraft full talk.mp4                           # 5 clips, default style
clipcraft full podcast.mp4 -n 3 --style tiktok    # 3 clips, TikTok style
clipcraft full video.mp4 --min-len 20 --max-len 60 # Shorter clips
clipcraft full video.mp4 --transcribe-only         # Just transcribe, stop there
clipcraft full video.mp4 --analyze-only            # Transcribe + analyze, no export
clipcraft full video.mp4 --no-resume               # Force fresh run (ignore cache)
```

### `clipcraft transcribe` — Transcription Only

```bash
clipcraft transcribe video.mp4                     # Local whisper (free)
clipcraft transcribe video.mp4 --api               # OpenAI API (paid, faster)
clipcraft transcribe video.mp4 -m large -l en      # Large model, force English
```

### `clipcraft find` — Clip Discovery

```bash
clipcraft find video.mp4                           # Auto-transcribes first
clipcraft find video.mp4 -t transcript.json        # Use cached transcript
clipcraft find video.mp4 -n 10 --min-len 15        # 10 clips, shorter minimum
```

### `clipcraft extract` — Export Clips

```bash
clipcraft extract video.mp4 -c clips.json -t transcript.json          # All clips
clipcraft extract video.mp4 -c clips.json -t transcript.json --clip 1 # Clip #1 only
clipcraft extract video.mp4 -c clips.json -t transcript.json -s tiktok
```

### `clipcraft styles` — Caption Styles

```bash
clipcraft styles
```

Available styles:
- **default** — Clean white text with gold highlight
- **tiktok** — Bold text with pink highlight, boxed background
- **minimal** — Small text at bottom, subtle
- **karaoke** — Gray base text, green word-by-word sweep

### `clipcraft info` — Video Metadata

```bash
clipcraft info video.mp4
# Duration: 1842s | Size: 450 MB | 1920×1080, h264, 30fps
```

## Configuration

Set these environment variables:

```bash
# For LLM analysis (required)
export OPENROUTER_API_KEY="sk-or-..."   # OpenRouter (default)
# or
export OPENAI_API_KEY="sk-..."           # OpenAI fallback

# Optional: override default model
export CLIPCRAFT_MODEL="anthropic/claude-sonnet-4"  # Default
# export CLIPCRAFT_MODEL="openai/gpt-4o"            # Alternative
```

## Caching

ClipCraft caches intermediate results so you don't pay twice:

- `*_transcript.json` — Reuse across multiple clip runs
- `*_clips.json` — Regenerate clips without re-transcribing
- Final clips — Skipped if output file already exists

Delete cache files to force re-processing.

## What Makes a Good Clip?

The LLM scores moments on:

| Signal | Example |
|--------|---------|
| 🪝 **Strong hook** | "I lost $50,000 in one day and here's what I learned" |
| 😲 **Surprising insight** | Counter-intuitive facts, revelations |
| 😤 **Emotional peak** | Anger, excitement, vulnerability |
| 🛠️ **Actionable advice** | Tactical tips the viewer can use now |
| ⚡ **Polarizing take** | Opinions that spark debate |
| 📖 **Great story** | Setup → conflict → resolution |

Each clip gets a 1-10 viral potential score.

## Cost

| Step | Cost |
|------|------|
| Transcription | **Free** (local faster-whisper) |
| LLM Analysis | ~$0.01-0.05 per video (Claude Sonnet via OpenRouter) |
| Clip Extraction | Free (FFmpeg, local) |

Processing a 60-minute video typically costs **under $0.05**.

## Architecture

```
clipcraft/
├── cli.py          # Click CLI with rich output
├── transcribe.py   # Whisper transcription (local + API)
├── analyze.py      # LLM clip analysis
├── extract.py      # FFmpeg clip extraction
├── caption.py      # ASS subtitle burning (4 styles)
├── pipeline.py     # End-to-end orchestration
```

## Roadmap

- [ ] Multi-speaker diarization (who said what)
- [ ] Auto-generated thumbnail frames
- [ ] Emoji overlay support
- [ ] Progress bar for transcription
- [ ] Batch processing (folder of videos)
- [ ] Direct upload to TikTok/Reels/Shorts
- [ ] Hermes Agent skill integration

## Why Open Source?

The video clipping market is dominated by SaaS tools ($20-200/mo) that lock you into their platform. ClipCraft is:

- **Free to run** — local whisper, your own LLM API key
- **You own the pipeline** — no vendor lock-in, no upload limits
- **Composable** — use it in scripts, CI/CD, cron jobs, Hermes agents
- **Private** — your content never leaves your machine (except the LLM API call if you use one)

Built for creators who want control over their content pipeline.

---

MIT License. Made with ❤️ by [@DBirker78883](https://x.com/DBirker78883)
