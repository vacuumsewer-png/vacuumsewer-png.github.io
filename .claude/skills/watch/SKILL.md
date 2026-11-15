---
name: watch
description: Watch a video for the user - YouTube, Loom, TikTok, Vimeo or any other site yt-dlp supports (well over 1,000), or a local video file. Gives Claude a timestamped transcript plus selected frames so it can see what's on screen, not just read captions. Use whenever the user pastes a video link or file and asks to watch, summarise, take notes on, or answer questions about it.
---

# watch

Turns a video into `index.md`: a timeline that interleaves transcript paragraphs with
frame image paths. Read that file, then open the frames with the Read tool to see them.

## Requirements

- `ffmpeg` and `ffprobe` on PATH
- `pip install yt-dlp pillow numpy` (yt-dlp is only needed for URLs)

## Modes (pick by need, default `efficient`)

| Mode | What it does | Use when |
|---|---|---|
| `transcript` | Captions only, no download of video | Talk/podcast, user wants content not visuals; fastest |
| `efficient` | Pulls the keyframes already stored in the file (no full decode), drops near-duplicates, max 40 frames | Everyday default |
| `balanced` | Scene-change detection + frames at "as you can see / look here" moments + 12 evenly spaced frames, deduped, max 80 | Slides, demos, screen-shares; visuals matter |
| `burner` | Every scene change up to 2 fps, no cap | Only when the user explicitly asks for everything; can produce hundreds of frames (expensive) |

## Run

```bash
python3 <this-skill-folder>/scripts/watch.py "<url-or-file>" --mode efficient --out ./watch_out
```

`<this-skill-folder>` is the folder holding this SKILL.md (in this repo:
`.claude/skills/watch`). Options: `--max-frames N` overrides the cap,
`--scene 0.2` makes scene detection more sensitive (default 0.3), `--lang xx` sets the
caption language. For a local file, captions are picked up from a sidecar
`<same-name>.vtt` if present.

The script prints one JSON line: `{"index": ..., "frames": N, "paragraphs": N, "duration": ...}`.

## Then

1. Read `index.md`.
2. Read the frame images. With more than ~40 frames, read the ones near the parts the
   user cares about first (frames marked `(deictic)` are where the speaker pointed at
   something on screen).
3. Answer with timestamps (`[00:12:34]`) so the user can jump to the moment.
4. Say which mode you used and how many frames you looked at. If no captions were found,
   say the answer is based on frames only.

## Limits

- No speech-to-text fallback yet: videos without captions (many Looms, raw local files)
  get frames only.
- Private or login-only videos need yt-dlp cookies (`--cookies-from-browser`), not wired in.
- `--out` is wiped at the start of each run.
