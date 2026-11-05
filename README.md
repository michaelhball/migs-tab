# migs-tab

Convert a YouTube acoustic guitar tutorial video into a guitar tab and a Markdown document of teaching tips.

Given a YouTube URL, `migs-tab` will:

1. Download the video, audio, and captions.
2. Isolate the guitar from any backing using [Demucs](https://github.com/facebookresearch/demucs).
3. Transcribe the isolated guitar into note events with two models: [YourMT3+](https://github.com/mimbres/YourMT3) (drives the tab) and [Spotify basic-pitch](https://github.com/spotify/basic-pitch) (ornament hints and cross-checks).
4. Identify the song's playing segments and their chord progressions, and detect the tuning and capo.
5. Track the fretting hand through the video (MediaPipe) and fit the fretboard's strings and fret wires in every sampled frame, so each fingertip can be read as a (string, fret) position.
6. Assign each note to a string and fret with a Viterbi search that scores playability, chord shapes, and where the fingers actually are. A note the transcriber heard an octave off is read at the octave the fingers support. Each fingertip presses one place (the index finger may barre), so one fingertip cannot make several notes look fingered. A note that sits on an overtone of a lower note in the same strum and has no fingertip near any placement is left out and listed in `frets.json` under `unfingered_overtones`. Where the camera does not show the hand, each pitch goes where the hand was seen playing it elsewhere in the video.
7. Score the result against the audio (`verify`) and, where reference passages exist, against known fingerings (`eval`).
8. Write a section-by-section ASCII tab, a Markdown wrapper, a chord chart and MusicXML, and (with the Claude skill described below) a tips document extracted from the instructor's commentary.

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
uv run migs-tab download       <url>   # video + audio + auto-captions
uv run migs-tab separate       <url>   # Demucs guitar isolation
uv run migs-tab transcribe     <url>   # basic-pitch note transcription
uv run migs-tab transcribe-mt3 <url>   # YourMT3+ note transcription
uv run migs-tab structure      <url>   # playing segments + chord progressions
uv run migs-tab tuning         <url>   # tuning + capo detection
uv run migs-tab handtrack      <url>   # fretting-hand landmarks per frame (hand_track.json)
uv run migs-tab fretboard      <url>   # per-frame strings + fret wires (fretboard.json)
uv run migs-tab frets          <url>   # string + fret assignment (uses the hand evidence)
uv run migs-tab verify         <url>   # audio-evidence scoring of the frets
uv run migs-tab eval           <url>   # score against eval/<id>.json reference passages
uv run migs-tab render         <url>   # write the tab
uv run migs-tab status         <url>   # show which artifacts are cached
uv run migs-tab doctor                 # preflight: binaries, packages, models, cache freshness
```

`migs-tab fretboard <url> --overlay-at 180.5,62.5` draws the fitted geometry and the tracked fingertips on those frames (under `cache/<id>/frames/fretboard/`), the quickest way to check the video step by eye.

### Checking accuracy

`eval/<video_id>.json` lists short passages whose fingering is known, usually because the instructor states it in the captions, as a vocabulary of allowed positions (`"A2"` = A string, fret 2) and optionally an ASCII tab of the passage. `uv run migs-tab eval <url>` reports, per passage, the fraction of the pipeline's notes at an allowed position, the fraction at an allowed pitch, how many allowed positions were used, and the number of notes far above the passage's highest fret. The reference files are hand-written and can be wrong; treat a passage's `confidence` field as a hint of how much to trust it.

```
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
│   ├── notes.mid, notes.json     basic-pitch transcription
│   ├── notes.mt3.mid, .json      YourMT3+ transcription (drives the tab)
│   ├── structure.json            playing segments + chord progressions
│   ├── tuning.json               tuning + capo
│   ├── hand_track.json           MediaPipe hand landmarks, 5 fps
│   ├── fretboard.json            per-frame string lines + fret-wire ladder
│   ├── sections.json             (produced by the Claude skill)
│   ├── frets.json                string/fret per note (+ octave corrections)
│   ├── verification.json         audio-evidence scores
│   └── tips.md                   (produced by the Claude skill)
├── eval/<video_id>.json          hand-written reference passages for `migs-tab eval`
└── output/<title>-<video_id>/    final tab files (gitignored)
    ├── tab.txt, tab.md
    ├── chord-chart.txt
    └── tab.musicxml
```

## Limitations

- Acoustic single-guitar tutorials only. Electric guitar with effects and multi-guitar arrangements are out of scope.
- The video step needs the usual tutorial framing: a mostly static camera in front of the player with the fretboard in view. It copes with the guitar drifting a few tens of pixels (geometry is fitted per frame) and with cuts between close-ups and wide shots (a wide shot gets no geometry, so onsets in it are decoded from audio alone; Justin Guitar's Song for George fits 60% of its frames this way). A hand-held or moving camera gets no geometry at all, and a capo makes the capo read as the nut.
- Tuning and capo are detected from the captions and the audio; detection can be wrong on songs that never use the lowest string.
- String choice for notes the fingers do not disambiguate (a chord strummed while the hand is off camera, open strings) falls back to playability heuristics.
- The transcription step depends on auto-captions being available on the YouTube video for the best `tips.md` and `sections.json` output. If captions are missing the audio analysis still works, but the LLM-driven section labeling will be less precise.
