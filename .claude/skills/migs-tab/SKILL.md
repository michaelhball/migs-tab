---
name: migs-tab
description: Convert a YouTube acoustic guitar tutorial URL into an accurate guitar tab plus a Markdown document of teaching tips and style notes. Use when the user provides a YouTube URL of a guitar tutorial and asks for a tab, tips, transcription, chord chart, or song structure analysis from it. Orchestrates a local Python CLI (yt-dlp, Demucs, basic-pitch, librosa) for the heavy lifting and uses Claude (this session) for synthesis steps.
---

# migs-tab — guitar tutorial → tab + tips

You are running inside the `migs-tab` project (`/Users/michaelball/projects/migs-tab`). The Python CLI handles all audio plumbing; you handle all LLM/synthesis steps. The user pays for Claude via subscription, so do not call any external LLM API — do the synthesis yourself, in this session.

## Inputs

The user provides a YouTube URL (or 11-char video ID). Extract the video ID and use it as the cache key throughout.

## Pipeline

Each step is idempotent — it skips work if its output already exists in the cache, unless `--force` is passed. Re-use cached artifacts aggressively; do not re-download or re-process anything that's already on disk.

### Step 1 — Plumbing (Python)

```bash
uv run migs-tab process <url>
```

Equivalent to running `download`, `separate`, `transcribe`, and `structure` in sequence. This populates `cache/<video_id>/` with:

- `video.mp4`, `audio.wav` — source media
- `captions.en.vtt`, `captions.txt` — auto-captions (full + flattened)
- `info.json` — title, uploader, duration, description
- `stems/other.wav` — guitar-isolated stem (Demucs)
- `notes.mid`, `notes.json` — polyphonic note events (basic-pitch)
- `structure.json` — playing segments + per-segment chord progressions + per-segment captions (librosa)

Notes:
- Demucs on a long video takes minutes on CPU. Be patient and let it finish.
- For quick iteration on a long video: `uv run migs-tab clip <url> --start 90 --duration 60 --name clip`, then `uv run migs-tab separate <url> --audio-name clip` — produces `stems/other.clip.wav` you can inspect.
- `--force` re-runs a step that's already cached.
- If `ffmpeg` or any binary is missing, surface the error verbatim — do not guess workarounds.

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

### Step 3 — Section labeling (you do this)

Read `cache/<id>/structure.json`. Produce `cache/<id>/sections.json` that labels each playing segment with its role in the song and groups repetitions of the same role.

**What `structure.json` contains:**
- `audio_duration` — total length of the audio in seconds.
- `playing_segments` — a list, each entry has:
  - `id`, `start`, `end`, `duration`
  - `rms_mean` — average loudness during the segment.
  - `chords` — coarse chord progression: list of `{chord, start, end}` spans within the segment. The chord names are major (`C`), minor (`Cm`), or dominant-seventh (`C7`) triads. The detector is good at distinguishing chord roots but can mistake `Am` for `A` or vice versa around chord voicings with added tones (E7 voicings, sus chords). Treat the progression as a rough fingerprint, not ground truth.
  - `captions` — the auto-captions said during or shortly before the segment.

**Your job:**

For each playing segment, decide what part of the song the instructor is demonstrating. Use BOTH the chord fingerprint AND the captions the instructor said leading into the segment (e.g., if they say "ok let's do the chorus" right before playing, that segment is the chorus). The captions are usually the strongest signal because instructors announce what they're about to demo.

Group segments that demonstrate the same part of the song. A tutorial usually plays the intro 4-6 times, the verse 3-4 times, etc. — clustering these gives the user multiple takes on each section, which Phase 3/4 will use to produce a more accurate canonical tab.

**`sections.json` format:**

```json
{
  "video_id": "<id>",
  "structural_summary": "<one-paragraph description of the song's structure and how the instructor walks through it>",
  "sections": [
    {
      "label": "intro",
      "description": "Brief description of musical content (e.g., 'A minor → E7 → G arpeggio over 4 bars').",
      "chord_progression": ["Am", "E7", "G", "Csus"],
      "instances": [
        {
          "segment_id": 4,
          "start": 95.2,
          "end": 128.6,
          "demo_quality": "slow-walkthrough | normal-tempo | repeated-loop | partial",
          "notes": "Optional: anything unusual about this take — e.g. 'slowest take, cleanest reference'."
        }
      ]
    }
  ],
  "unclassified_segments": [
    { "segment_id": N, "reason": "why you couldn't confidently label this" }
  ]
}
```

Section labels you might use (pick what fits the song; the instructor may use different vocabulary):
- `intro`, `verse_1`, `verse_2`, `chorus`, `pre_chorus`, `bridge`, `outro`, `solo`, `lick_<name>`, `chord_demo_<name>`, `transition_<name>`.

Only put a segment in `unclassified_segments` if you genuinely can't tell what it's demonstrating from the chord fingerprint and the captions. Don't guess wildly.

After writing sections.json, print a brief summary to the user: total segments, how many got labeled, how many distinct sections, and an example of one section with its instance count (e.g., "intro: 5 instances spanning 1:32 to 18:04").

### Step 4 — Sanity output

After Step 3, print a final summary:
- Video title + duration + uploader
- Note count from `notes.json`
- Playing-segment count from `structure.json`
- Distinct section count from `sections.json`
- File paths produced
- The first ~150 chars of `tips.md` as a preview

Do not dump the raw caption transcript, notes.json, or structure.json content.

## Phase 3-4 (not yet implemented)

When the user asks for these, tell them they're future work:
- Phase 3: heuristic + vision-pass fret optimization, picking exact fret/string positions per note.
- Phase 4: ASCII tab + MusicXML rendering using the grouped instances from sections.json for redundancy.

See `SPEC.md` for the full design.

## Failure modes you should handle

- **No captions available** — yt-dlp will not produce `captions.en.vtt`. Run the plumbing, then write a tips.md noting captions weren't available, and label sections from chord fingerprints alone. The audio analysis still works.
- **Demucs hangs / slow** — first run downloads model weights (~80MB). Subsequent runs are faster. Do not kill the process; let it finish.
- **basic-pitch produces zero notes** — likely a bad/silent stem. Inspect with `ffprobe stems/other.wav` and report.
- **The wrong stem is captured** — if the "other" stem doesn't sound like the guitar (e.g., backing track has prominent piano), try `--model htdemucs_ft` (slower, fine-tuned) and rerun separate with `--force`.
- **Chord progressions look like nonsense** — the detector struggles when the instructor is mostly arpeggiating single notes with no clear chord context. Trust captions more than chord progression in these segments.
- **Way too many playing segments (or way too few)** — sensitivity is set by `top_db` in `structure.py`. Re-run with `--force` after adjusting if needed.

## Conventions

- Always run CLI commands with `uv run` so they use the project venv.
- Work relative to the project root `/Users/michaelball/projects/migs-tab`.
- Cache is at `cache/<video_id>/`. Outputs land in the cache dir. Final user-facing outputs (when Phase 4 lands) go to `output/<video_id>/`.
- Reuse cached artifacts. Never re-run a step whose output is on disk unless the user asks for `--force` or you've changed inputs.
