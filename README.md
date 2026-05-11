# migs-tab

Convert a YouTube acoustic guitar tutorial video into a guitar tab and a Markdown document of teaching tips.

Given a YouTube URL, `migs-tab` will:

1. Download the video, audio, and captions.
2. Isolate the guitar from any backing using [Demucs](https://github.com/facebookresearch/demucs).
3. Transcribe the isolated guitar into a sequence of notes using [Spotify basic-pitch](https://github.com/spotify/basic-pitch).
4. Identify the song's playing segments and their chord progressions.
5. Assign each note to a string and fret on a standard-tuned 6-string guitar.
6. Write the results to disk as a section-by-section ASCII tab plus a Markdown wrapper, and (when paired with the Claude skill described below) a tips document extracted from the instructor's commentary.

The tab is intended for personal use — to help you learn songs from instructional videos that scatter teaching across many short demonstrations.

## What you need

- A Mac or Linux machine.
- Python 3.11 (installed automatically by `uv`).
- [`uv`](https://github.com/astral-sh/uv) for managing Python and dependencies. Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- `ffmpeg` on your `PATH`. On macOS: `brew install ffmpeg`.
- About 1 GB of free disk space per processed video for cached audio, video, and intermediate artifacts. A 25-minute tutorial typically takes 15–20 minutes of CPU time on a modern laptop for the audio separation step.

No GPU is required. No API keys are required for the CLI itself.

## Installing

```bash
git clone git@github.com:michaelhball/migs-tab.git
cd migs-tab
uv sync
```

`uv sync` will install Python 3.11 and all dependencies into a project-local virtual environment.

## Using it

### Whole pipeline in one command

```bash
uv run migs-tab process "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

This downloads the video, isolates the guitar, transcribes the notes, analyzes the song's structure, and assigns frets — all idempotent and cached under `cache/<VIDEO_ID>/`. The slowest step is guitar separation, which takes about 1.5–2× the video's length on CPU.

When the pipeline finishes, render the tab:

```bash
uv run migs-tab render "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

Outputs land at:

- `output/<VIDEO_ID>/tab.txt` — section-by-section ASCII tab.
- `output/<VIDEO_ID>/tab.md` — the same content as Markdown, with the song's structural summary at the top.

### Running individual steps

Each step is idempotent — it re-uses the cache unless you pass `--force`.

```bash
uv run migs-tab download   <url>   # video + audio + auto-captions
uv run migs-tab separate   <url>   # Demucs guitar isolation
uv run migs-tab transcribe <url>   # basic-pitch note transcription
uv run migs-tab structure  <url>   # playing segments + chord progressions
uv run migs-tab frets      <url>   # string + fret assignment
uv run migs-tab render     <url>   # write the tab
uv run migs-tab status     <url>   # show which artifacts are cached
```

### Quick iteration on a short audio segment

If you want to test the audio-analysis pipeline on a short clip of a long video, slice with `clip`:

```bash
uv run migs-tab clip <url> --start 90 --duration 60 --name clip
uv run migs-tab separate <url> --audio-name clip
```

This produces a `stems/other.clip.wav` you can listen to.

## Using it from Claude Code (recommended)

The repo includes a Claude Code skill at `.claude/skills/migs-tab/SKILL.md` that orchestrates the whole pipeline and also produces:

- `cache/<VIDEO_ID>/tips.md` — teaching tips and style notes distilled from the tutorial's captions.
- `cache/<VIDEO_ID>/sections.json` — the song's playing segments labeled with their role in the song (intro, verse, chorus, bridge, outro), with repeated demonstrations grouped together.

When you open this project in Claude Code, type:

```
/migs-tab https://www.youtube.com/watch?v=<VIDEO_ID>
```

Claude will run the CLI for the audio plumbing and use its own reasoning (against your Claude subscription) for the steps that need natural-language understanding. No API key is needed.

## Where the files go

```
migs-tab/
├── cache/<video_id>/             intermediate artifacts (gitignored)
│   ├── video.mp4
│   ├── audio.wav
│   ├── captions.en.vtt
│   ├── captions.txt
│   ├── stems/other.wav           guitar-isolated stem
│   ├── notes.mid                 transcribed notes
│   ├── notes.json
│   ├── structure.json
│   ├── sections.json             (produced by the Claude skill)
│   ├── frets.json
│   └── tips.md                   (produced by the Claude skill)
└── output/<video_id>/            final tab files (gitignored)
    ├── tab.txt
    └── tab.md
```

## Limitations

- Acoustic single-guitar tutorials only. Electric guitar with effects, multi-guitar arrangements, and alternate tunings are out of scope for now.
- The CLI assumes standard tuning (E A D G B E).
- The transcription step depends on auto-captions being available on the YouTube video for the best `tips.md` and `sections.json` output. If captions are missing the audio analysis still works, but the LLM-driven section labeling will be less precise.
