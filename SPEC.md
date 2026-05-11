# migs-tab — Spec

Local CLI tool that converts a YouTube acoustic-guitar-tutorial video into a publishable-accuracy tab plus a separate document of tips/style notes extracted from the instructor's commentary.

## Goals & non-goals

**Goals**
- Acoustic, single-guitar tutorials (the goal — not multi-instrument, not electric with FX).
- Output suitable for the user to learn the song from. "Publishable accuracy", not "good enough to play along by ear".
- Runs entirely locally (Demucs, basic-pitch) with the exception of Claude API calls for synthesis steps.
- Combines repeated sections of the video into a single canonical tab (intro / verse / chorus / etc.).
- Produces a separate tips-and-style-notes markdown document distilled from the spoken transcript.

**Non-goals (for now)**
- Electric guitar, fingerstyle multi-voice, alternate tunings.
- Real-time / streaming use.
- Hosted web app — local CLI only.
- Publishing tabs externally — personal use.

## Architecture

```
YouTube URL
   │
   ▼
[1] yt-dlp ────────────► video.mp4, audio.wav, captions.{vtt,json}
   │
   ▼
[2] Demucs ───────────► stems/{vocals,drums,bass,other}.wav  (we use "other" for guitar)
   │
   ▼
[3] basic-pitch ──────► notes.json (onset, offset, pitch, velocity), notes.mid
   │
   ▼
[4] Claude API
   ├─ tips extraction (from captions) ─► tips.md
   ├─ section detection / merge ──────► sections.json
   ├─ fret optimization (heuristic + vision) ─► tab.json
   └─ rendering ───────────────────────► tab.txt (ASCII), tab.musicxml
```

All intermediate artifacts are cached per-video under `cache/{video_id}/`, so re-running a phase is cheap and individual phases can be debugged in isolation.

## Phases

### Phase 1 — Foundation (this iteration)
- `yt-dlp` ingestion (video, audio, captions).
- Demucs stem isolation.
- basic-pitch transcription → raw note sequence + MIDI.
- Tips extraction from captions via Claude API → `tips.md`.
- Output at this stage: a MIDI file the user can play back to sanity-check transcription, plus the tips document. No structured tab yet.

### Phase 2 — Structure
- Detect repeated sections by aligning note sequences against themselves.
- Use Claude to cross-reference caption timestamps ("now the chorus", "intro again") with the audio-derived structure.
- Output: `sections.json` with canonical labelled segments + a section-tagged tab skeleton.

### Phase 3 — Fret optimization
- Heuristic pass: for each note, pick the fret/string combination that minimizes hand movement from the previous note (greedy + small lookahead). Constrain to standard tuning + reasonable hand position.
- Vision pass for ambiguous cases:
  - Identify notes where two or more fret positions have similar cost (e.g., open E vs 5th-fret B string).
  - Sample one or a few video frames at the note's onset time.
  - Batch frames into 3×3 grids; send to Claude with a prompt asking for fret/string position based on visible hand position.
  - Cache vision results.
- Output: `tab.json` with full fret/string assignments per note.

### Phase 4 — Rendering
- ASCII tab generator with section labels and timing/measure bars.
- MusicXML export for Guitar Pro / MuseScore import.

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 | ML library compat. Pinned via `.python-version`. |
| Package mgr | `uv` | Already installed; fast resolver; modern. |
| CLI | `typer` | Type-driven, ergonomic subcommands. |
| Download | `yt-dlp` | De-facto YouTube download lib. |
| Audio sep | `demucs` (PyTorch) | Open, local, no TF. Apple Silicon MPS support. |
| Transcription | `basic-pitch` w/ ONNX backend | Polyphonic, MIT, runs on CPU in seconds. No TF. |
| Captions | `youtube-transcript-api` | Fetches auto-generated captions when available. |
| MIDI I/O | `pretty_midi` | Easy MIDI read/write + per-note manipulation. |
| LLM | `anthropic` SDK | For tips, section detection, fret disambiguation, vision pass. |

## Project layout

```
migs-tab/
├── pyproject.toml
├── SPEC.md                          (this file)
├── README.md
├── src/migs_tab/
│   ├── __init__.py
│   ├── cli.py                       Typer entry point
│   ├── paths.py                     cache/output dir helpers + per-video ID dirs
│   ├── download.py                  yt-dlp wrapper
│   ├── separate.py                  Demucs wrapper
│   ├── transcribe.py                basic-pitch wrapper
│   ├── captions.py                  caption fetch + text-only flattening
│   └── tips.py                      Claude tips extraction
├── cache/                           per-video intermediate artifacts (gitignored)
└── output/                          final tab + tips files (gitignored)
```

## CLI surface (Phase 1)

```
migs-tab download <url>           # download video, audio, captions
migs-tab separate <url>           # run Demucs on cached audio
migs-tab transcribe <url>         # run basic-pitch on isolated guitar stem
migs-tab tips <url>               # extract tips via Claude API
migs-tab process <url>            # all of the above end-to-end
```

Each subcommand operates on a URL and uses the same cache layout, so partial reruns are cheap. `process` runs all four phases in order, skipping any step whose output already exists (unless `--force` is passed).

## Open questions / future

### Future: non-standard tuning support

The CLI currently hard-codes standard tuning `(E A D G B E)` everywhere. Supporting alternate tunings (drop D, DADGAD, half-step-down, open tunings, capo positions) is feasible but touches several layers. Sketch:

1. **Detection.** Instructors almost always announce the tuning explicitly. The Claude skill should parse the captions for phrases like *"tune your low E down to D"*, *"we're in DADGAD"*, *"capo on the 3rd fret"*, *"drop D"*, *"half-step down"*. Output a `tuning.json` in the cache dir:
   ```json
   {"strings_midi": [38, 45, 50, 55, 59, 64], "capo": 0, "label": "Drop D"}
   ```
2. **Plumb tuning through fret.py.** Replace the module-level `STANDARD_TUNING` constant with a per-video lookup. `assign_frets` reads `tuning.json` if present, else falls back to standard. The Viterbi already enumerates `fret = pitch - tuning[string]` per string, so changing `tuning` automatically reshapes the option space.
3. **Chord templates.** `_CHORD_TEMPLATES` is keyed by pitch → `(string, fret)` for standard tuning. Two options:
   - Recompute the templates dynamically from open-chord *shapes* (e.g., E open shape = `[0,2,2,1,0,0]` strings 0..5) for whatever tuning is loaded. This is the cleaner long-term approach.
   - Or maintain per-tuning template libraries (Drop-D set, DADGAD set, etc.) and pick based on `tuning.json`.
4. **Capo.** Pitches are absolute, so a capo at fret N effectively raises every open string's pitch by N semitones. Two ways:
   - Treat the capo as a virtual nut: tuning becomes `[t + N for t in standard]` and frets above the capo are relative.
   - Or keep absolute frets and just label "capo N" in the tab header so the player knows.
5. **Render.** Tab header shows the detected tuning + capo. The string letters `[e, B, G, D, A, E]` may need to change (e.g., for drop D the bottom line is `D`, not `E`).
6. **Caption-driven LLM step.** The skill should set the tuning *before* `migs-tab frets` runs, since fret assignment depends on it. So the order becomes: download → separate → transcribe → structure → (LLM reads captions and writes `tuning.json`) → frets → render.

Rough effort: ~half a day for steps 1-5 once we commit. Step 3 is the only nontrivial piece — dynamic chord-template generation requires writing chord *shapes* (string-relative fret offsets from the root) rather than the current pitch-keyed dictionaries.

### Other open items
- basic-pitch is general-purpose; if accuracy is insufficient on acoustic guitar, evaluate MT3 as the upgrade path (likely server-side because of GPU).
- For phase-3 vision: a small open-source fret-detection CNN (trained on GuitarSet) could handle the easy cases locally before escalating to Claude — TBD whether that's worth the engineering.
- The "other" stem from Demucs is a guitar proxy after vocal/drum/bass removal. If a tutorial has additional background music, isolation may be imperfect — consider htdemucs_ft (fine-tuned) or running Demucs in 6-source mode.
- MusicXML export for Guitar Pro / MuseScore / TuxGuitar.
