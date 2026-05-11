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
- `frets.json` — heuristic (string, fret) per note + ambiguous flag + alternatives per cluster (Viterbi)

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

### Step 4 — Fret optimization (Python heuristic, already done in plumbing)

`uv run migs-tab process` already runs the heuristic Viterbi over notes.json
and produces `cache/<id>/frets.json`, containing one (string, fret) per note
plus an `ambiguous` flag per cluster and a list of `alternatives`. You don't
need to do anything here unless the user asks you to re-tune the weights.

### Step 5 — Vision pass on ambiguous fret assignments (you do this — bounded)

**IMPORTANT — quota guardrail:** Each vision-pass frame is a Read on a JPEG,
which costs Claude tokens. Never extract more than the user is willing to pay
for. The default ceiling is **5 frames per pass** and **at most one pass per
song section** unless the user explicitly asks for a wider sweep. The CLI
enforces this via `--max-frames` (default 10, but for routine use prefer 3-5).

Workflow:

1. **Pick a tiny, high-value subset of ambiguous clusters.** Don't iterate over
   all clusters in frets.json — there are typically hundreds. Prefer:
   - Clusters where the *intrinsic-cost delta* between chosen and alternative
     is very small (near-tie cases where vision will make the biggest impact).
   - Clusters in canonical-reference segments from sections.json (a single
     correction on the slow intro replay propagates structurally across the
     other intro instances).
   - A few clusters spread across the song so we sample different hand
     positions rather than re-confirming the same passage.

2. **Extract frames** (CLI does this; clamps to max_frames):
   ```bash
   uv run migs-tab frames-for-clusters <url> <comma-separated-ids> \
       --max-frames 5 --subdir <descriptive_name>
   ```
   Writes `cache/<id>/frames/<subdir>/t<seconds>_cluster<id>.jpg`.

3. **Read each frame with the Read tool.** For each frame:
   - Identify which fret region the visible hand is in. The dot inlays on the
     fretboard are reliable anchors: single dots at frets 3, 5, 7, 9, then a
     double dot at fret 12 (on most acoustics).
   - Decide whether the algorithm's chosen (string, fret) is consistent with
     the visible hand position, or whether one of the listed `alternatives`
     fits better.
   - If the hand is mid-transition (no fingers pressed), abstain — do not
     override the algorithm. Note this in your output.

4. **Write `cache/<id>/frets.overrides.json`** with the corrections:
   ```json
   {
     "video_id": "<id>",
     "overrides": [
       {
         "cluster_id": 1813,
         "reason": "Hand clearly at fret 7 — algorithm picked fret 17.",
         "new_assignments": [{"note_index": 0, "string": 3, "fret": 7}]
       }
     ],
     "unchanged_after_review": [1838],
     "abstained": [{"cluster_id": 1900, "reason": "Hand mid-transition, no fingers pressed."}]
   }
   ```

5. **Print to the user**: how many frames reviewed, how many corrections, a
   one-line description of each correction. Do not show the frames themselves.

**Do NOT** sweep ambiguous clusters across the full song without the user's
explicit go-ahead. Default behavior is conservative — review a small, targeted
batch and report.

## Phase 4 (not yet implemented)

When the user asks for it, tell them it's future work:
- Phase 4: ASCII tab + MusicXML rendering. Uses sections.json to organize the
  tab by song role, drawing notes from the longest/cleanest instance of each
  section and applying `frets.json` + `frets.overrides.json` for fret choices.

See `SPEC.md` for the full design.

## Failure modes you should handle

- **No captions available** — yt-dlp will not produce `captions.en.vtt`. Run the plumbing, then write a tips.md noting captions weren't available, and label sections from chord fingerprints alone. The audio analysis still works.
- **Demucs hangs / slow** — first run downloads model weights (~80MB). Subsequent runs are faster. Do not kill the process; let it finish.
- **basic-pitch produces zero notes** — likely a bad/silent stem. Inspect with `ffprobe stems/other.wav` and report.
- **The wrong stem is captured** — if the "other" stem doesn't sound like the guitar (e.g., backing track has prominent piano), try `--model htdemucs_ft` (slower, fine-tuned) and rerun separate with `--force`.
- **Chord progressions look like nonsense** — the detector struggles when the instructor is mostly arpeggiating single notes with no clear chord context. Trust captions more than chord progression in these segments.
- **Way too many playing segments (or way too few)** — sensitivity is set by `top_db` in `structure.py`. Re-run with `--force` after adjusting if needed.
- **Vision pass returns ambiguous answers** — many tutorial frames catch the hand mid-transition with no fingers pressed; abstain rather than guess. If multiple frames around a cluster all look mid-transition, try extracting at onset - 0.1s by using `migs-tab frame <url> <onset_minus_0.1>` manually.

## Conventions

- Always run CLI commands with `uv run` so they use the project venv.
- Work relative to the project root `/Users/michaelball/projects/migs-tab`.
- Cache is at `cache/<video_id>/`. Outputs land in the cache dir. Final user-facing outputs (when Phase 4 lands) go to `output/<video_id>/`.
- Reuse cached artifacts. Never re-run a step whose output is on disk unless the user asks for `--force` or you've changed inputs.
