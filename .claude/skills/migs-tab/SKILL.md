---
name: migs-tab
description: Convert a YouTube acoustic guitar tutorial URL into an accurate guitar tab plus a Markdown document of teaching tips and style notes. Use when the user provides a YouTube URL of a guitar tutorial and asks for a tab, tips, transcription, or chord chart from it. Orchestrates a local Python CLI (yt-dlp, Demucs, basic-pitch) for the heavy lifting and uses Claude (this session) for synthesis steps.
---

# migs-tab — guitar tutorial → tab + tips

You are running inside the `migs-tab` project (`/Users/michaelball/projects/migs-tab`). The Python CLI handles all audio plumbing; you handle all LLM/synthesis steps. The user pays for Claude via subscription, so do not call any external LLM API — do the synthesis yourself, in this session.

## Inputs

The user provides a YouTube URL (or 11-char video ID). Extract the video ID and use it as the cache key throughout.

## Pipeline (Phase 1 — current scope)

Run each step only if its output is missing or the user asked you to re-run.

### Step 1 — Plumbing

```bash
uv run migs-tab process <url>
```

Equivalent to running `download`, `separate`, and `transcribe` in sequence. This populates `cache/<video_id>/` with:

- `video.mp4`, `audio.wav` — source media
- `captions.en.vtt`, `captions.txt` — auto-captions (flattened to plain text)
- `info.json` — title, uploader, duration, description
- `stems/other.wav` — guitar-isolated stem (Demucs)
- `notes.mid`, `notes.json` — polyphonic note events (basic-pitch)

Notes:
- Demucs on a long video takes minutes on CPU — be patient and stream output rather than polling.
- If the user just wants a quick test, you can clip first: `uv run migs-tab clip <url> --start 90 --duration 60 --name clip`, then `uv run migs-tab separate <url> --audio-name clip`.
- `--force` re-runs a step that's already cached.
- If `ffmpeg` or any binary is missing, surface the error verbatim to the user — do not guess workarounds.

### Step 2 — Tips extraction (you do this)

Read `cache/<id>/captions.txt` and `cache/<id>/info.json`. Produce `cache/<id>/tips.md` following the format below.

If captions.txt is empty or missing, write a one-line tips.md noting captions were unavailable. Do not invent content.

**Tips document format:**

```markdown
# Tips — <song title>

_Source: <uploader> • https://youtu.be/<video_id>_

## Summary

One sentence: what song, what artist, what key/tuning if mentioned, capo position if mentioned, approximate skill level if the instructor flags it.

## Key techniques

For each distinct technique the instructor teaches or demonstrates, one bullet:
- **<Technique name>** — one-line description of how/where in the song.

Examples of techniques to surface: specific strumming patterns, hammer-ons / pull-offs, palm muting, percussive strikes, fingerpicking patterns (Travis picking, etc.), harmonics, bends, vibrato, capo placement, alternate tunings, chord voicings the instructor flags as unusual.

## Tips & style notes

Bulleted advice from the instructor that goes beyond "play these notes" — feel, dynamics, when to breathe, where players commonly mess up, what to listen for in the original recording, performance suggestions.

Quote the instructor directly when a phrase is especially memorable. Use markdown italics for direct quotes: _"like this"_.

## Practice suggestions

Only include this section if the instructor explicitly gives practice advice (slow-down strategies, drills, "loop this until it's clean", etc.). If absent, omit the section entirely.
```

**Rules — non-negotiable:**

- Be faithful to the instructor. Do not infer technique from your knowledge of the song that the instructor did not actually mention.
- Skip greetings, sign-offs, sponsor reads, Patreon/like-subscribe asks, gear plugs.
- If the auto-captions are garbled in a particular line, omit it rather than guessing.
- Omit empty sections entirely (don't write "## Practice suggestions: none").

Write the file with the Write tool, not via the CLI.

### Step 3 — Sanity output

After Step 2, print a short summary to the user:
- Video title + duration + uploader
- Note count from `notes.json`
- File paths produced (cache dir + tips.md)
- The first ~150 chars of the tips.md as a preview

Do not show the raw caption transcript or the full notes.json contents.

## Phase 2-4 (not yet implemented)

These are stubs — when the user asks for them, tell them they're future work:
- Phase 2: section detection + repetition merging from `notes.json` cross-referenced with caption timestamps.
- Phase 3: fret optimization (heuristic + vision pass on extracted frames).
- Phase 4: ASCII tab + MusicXML rendering.

See `SPEC.md` for the full design.

## Failure modes you should handle

- **No captions available** — yt-dlp will not produce `captions.en.vtt`. Run the plumbing, then write a tips.md noting that captions weren't available. The audio analysis still works.
- **Demucs hangs / slow** — first run downloads model weights (~80MB). Subsequent runs are faster. Do not kill the process; let it finish.
- **basic-pitch produces zero notes** — likely a bad/silent stem. Inspect with `ffprobe stems/other.wav` and report.
- **The wrong stem is captured** — if the "other" stem doesn't sound like the guitar (e.g., backing track has prominent piano), try `--model htdemucs_ft` (slower, fine-tuned) and rerun separate with `--force`.

## Conventions

- Always run CLI commands with `uv run` so they use the project venv.
- Work relative to the project root `/Users/michaelball/projects/migs-tab`.
- Cache is at `cache/<video_id>/`. Outputs land in the cache dir. Final user-facing outputs (when Phase 4 lands) go to `output/<video_id>/`.
