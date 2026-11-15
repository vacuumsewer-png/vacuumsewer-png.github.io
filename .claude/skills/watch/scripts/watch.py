#!/usr/bin/env python3
"""watch.py - turn a video (URL or local file) into a timestamped transcript
plus a small set of meaningful frames that Claude can read.

Modes:
  transcript  captions only, no frames (fastest)
  efficient   keyframes already stored in the video (no full decode), deduped, capped
  balanced    scene-change frames + "as you can see" (deictic) moments + even fill, deduped, capped
  burner      every scene change, up to 2 fps, no cap (expensive)

Output: <out>/index.md (frames interleaved with transcript), <out>/transcript.md,
        <out>/frames/*.jpg  (filenames carry the timestamp)

Requires: ffmpeg/ffprobe on PATH, yt-dlp (for URLs), Pillow, numpy.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image

CAPS = {"efficient": 40, "balanced": 80, "burner": None}
DEICTIC = re.compile(
    r"\b(as you can see|you can see|look (here|at this|at that)|see (here|this|that)|"
    r"watch (this|that)|right here|over here|on (the )?screen|this (one|slide|part)|"
    r"what happens|notice (how|that)|take a look|check (this|that) out)\b", re.I)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def log(*a):
    print("[watch]", *a, file=sys.stderr)


def ts(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- acquire
def acquire(src, work, need_video, lang):
    """Return (video_path or None, caption_path or None, title)."""
    if os.path.exists(src):
        side = os.path.splitext(src)[0] + ".vtt"  # optional sidecar captions for local files
        return os.path.abspath(src), (side if os.path.exists(side) else None), os.path.basename(src)
    if not shutil.which("yt-dlp"):
        sys.exit("yt-dlp not found: pip install yt-dlp")
    base = ["yt-dlp", "--no-playlist", "--no-warnings", "-o", os.path.join(work, "src.%(ext)s")]
    # exact codes only: "en.*" would also match auto-translated tracks (en-de, ...) and trip 429s
    subs = ["--write-subs", "--write-auto-subs", "--sub-langs",
            f"{lang},{lang}-orig,{lang}-US,{lang}-GB", "--sub-format", "vtt/best", "--convert-subs", "vtt"]
    meta = run(["yt-dlp", "--no-playlist", "-J", src])
    title = src
    if meta.returncode == 0:
        try:
            title = json.loads(meta.stdout).get("title") or src
        except json.JSONDecodeError:
            pass
    # captions and video in separate calls so a caption error never blocks the video
    r = run(base + subs + ["--skip-download", src])
    if r.returncode != 0:
        log("caption download problem:", r.stderr.strip()[-300:])
    if need_video:
        r = run(base + ["-f", "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
                        "--merge-output-format", "mp4", src])
        if r.returncode != 0:
            sys.exit("yt-dlp video download failed:\n" + r.stderr[-2000:])
    vids = [p for p in glob.glob(os.path.join(work, "src.*"))
            if not p.endswith((".vtt", ".srt", ".json", ".part"))]
    caps = sorted(glob.glob(os.path.join(work, "src*.vtt")),
                  key=lambda p: (".orig" in p, "auto" in p, len(p)))  # prefer manual/shorter names
    return (vids[0] if vids else None), (caps[0] if caps else None), title


# ---------------------------------------------------------------- transcript
def parse_vtt(path):
    """VTT -> list of (start_sec, text), with YouTube rolling-caption duplicates removed."""
    out, last = [], ""
    t = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        m = re.match(r"(\d+):(\d\d):(\d\d)[.,](\d+)\s+-->", line) or \
            re.match(r"()(\d\d):(\d\d)[.,](\d+)\s+-->", line)
        if m:
            h = int(m.group(1) or 0)
            t = h * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            continue
        if t is None or not line or line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if not text or text == last or (out and text in out[-1][1]):
            continue
        last = text
        out.append((t, text))
    return out


def merge_lines(lines, window=15):
    """Group caption lines into ~window-second paragraphs."""
    paras, cur, start = [], [], None
    for t, txt in lines:
        if start is None:
            start = t
        if t - start >= window and cur:
            paras.append((start, " ".join(cur)))
            cur, start = [], t
        cur.append(txt)
    if cur:
        paras.append((start, " ".join(cur)))
    return paras


# ---------------------------------------------------------------- frames
def duration(video):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def extract_with_pts(video, outdir, vf, pre=(), prefix="f"):
    """Run ffmpeg with a filter, write JPGs, return [(sec, path)] using showinfo pts."""
    os.makedirs(outdir, exist_ok=True)
    pattern = os.path.join(outdir, f"{prefix}_%05d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", *pre, "-i", video,
           "-vf", vf + ",showinfo", "-fps_mode", "vfr", "-q:v", "4", pattern]
    r = run(cmd)
    times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    files = sorted(glob.glob(os.path.join(outdir, f"{prefix}_*.jpg")))
    return list(zip(times, files))


def grab_at(video, outdir, secs, prefix="d"):
    os.makedirs(outdir, exist_ok=True)
    out = []
    for i, s in enumerate(secs):
        p = os.path.join(outdir, f"{prefix}_{i:05d}.jpg")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{s:.2f}", "-i", video,
             "-frames:v", "1", "-vf", "scale='min(960,iw)':-2", "-q:v", "4", "-y", p])
        if os.path.exists(p):
            out.append((s, p))
    return out


def thumb(path):
    return np.asarray(Image.open(path).convert("L").resize((24, 24)), dtype=np.float32)


def dedupe(frames, thresh):
    """Drop frames that look almost the same as the last kept one (mean abs diff on 24x24 gray)."""
    kept, last = [], None
    for s, p in sorted(frames):
        th = thumb(p)
        if last is not None and np.abs(th - last).mean() < thresh:
            continue
        kept.append((s, p))
        last = th
    return kept


def cap(frames, n, must=()):
    """Keep at most n frames, evenly spread; frames whose time is in `must` are preferred."""
    if n is None or len(frames) <= n:
        return frames
    must = set(must)
    pri = [f for f in frames if f[0] in must][: n // 2]
    rest = [f for f in frames if f not in pri]
    k = n - len(pri)
    idx = np.linspace(0, len(rest) - 1, k).round().astype(int) if k > 0 else []
    return sorted(pri + [rest[i] for i in sorted(set(idx))])


def select_frames(video, mode, work, lines, scene, max_frames):
    raw = os.path.join(work, "raw")
    dur = duration(video)
    scale = "scale='min(960,iw)':-2"
    deictic = []
    if mode == "efficient":
        frames = extract_with_pts(video, raw, scale, pre=["-skip_frame", "nokey"], prefix="k")
        frames = dedupe(frames, 6.0)
    else:
        fps = ",fps=2" if mode == "burner" else ""
        frames = extract_with_pts(video, raw, f"select='gt(scene,{scene})'{fps},{scale}", prefix="s")
        if mode == "balanced":
            deictic = sorted({t + 1 for t, txt in lines if DEICTIC.search(txt)})
            deictic = deictic[:: max(1, len(deictic) // 40)]  # bound ffmpeg seeks on long talks
            frames += grab_at(video, raw, deictic, prefix="d")
            n_fill = 12 if dur > 0 else 0  # even coverage so quiet stretches aren't blind
            frames += grab_at(video, raw, [dur * (i + 0.5) / n_fill for i in range(n_fill)], prefix="e")
        frames += grab_at(video, raw, [0.0], prefix="z")  # always include the opening frame
        frames = dedupe(frames, 4.0 if mode == "burner" else 6.0)
    limit = max_frames if max_frames is not None else CAPS[mode]
    frames = cap(frames, limit, must=deictic)
    # copy kept frames to frames/ with timestamped names
    fdir = os.path.join(work, "frames")
    os.makedirs(fdir, exist_ok=True)
    out = []
    for s, p in frames:
        dst = os.path.join(fdir, f"t{int(s):06d}_{ts(s).replace(':', '')}.jpg")
        if not os.path.exists(dst):
            shutil.copy(p, dst)
            out.append((s, dst, s in set(deictic)))
    shutil.rmtree(raw, ignore_errors=True)
    return out, dur


# ---------------------------------------------------------------- output
def write_outputs(work, title, src, mode, paras, frames, dur):
    with open(os.path.join(work, "transcript.md"), "w") as f:
        f.write(f"# Transcript: {title}\n\n")
        for t, txt in paras:
            f.write(f"[{ts(t)}] {txt}\n\n")
    events = [(t, 1, f"[{ts(t)}] {txt}") for t, txt in paras]
    events += [(s, 0, f"FRAME {ts(s)}{' (deictic)' if d else ''}: {p}") for s, p, d in frames]
    with open(os.path.join(work, "index.md"), "w") as f:
        f.write(f"# Watch: {title}\n\nSource: {src}\nMode: {mode} | Duration: {ts(dur)} | "
                f"Frames: {len(frames)} | Transcript paragraphs: {len(paras)}\n\n")
        if not paras:
            f.write("> No captions found. Frames only.\n\n")
        for _, _, line in sorted(events):
            f.write(line + "\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video URL or local file path")
    ap.add_argument("--mode", choices=["transcript", "efficient", "balanced", "burner"], default="efficient")
    ap.add_argument("--out", default=None, help="output dir (default: ./watch_out)")
    ap.add_argument("--max-frames", type=int, default=None, help="override frame cap")
    ap.add_argument("--scene", type=float, default=0.3, help="scene-change threshold 0-1 (balanced/burner)")
    ap.add_argument("--lang", default="en", help="caption language code")
    a = ap.parse_args()

    work = os.path.abspath(a.out or "watch_out")
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    need_video = a.mode != "transcript"
    video, capfile, title = acquire(a.source, work, need_video, a.lang)
    lines = parse_vtt(capfile) if capfile else []
    if not lines:
        log("no captions found (speech-to-text fallback is not implemented yet)")
    paras = merge_lines(lines)
    frames, dur = [], 0.0
    if need_video:
        if not video:
            sys.exit("no video file downloaded")
        frames, dur = select_frames(video, a.mode, work, lines, a.scene, a.max_frames)
        if not a.source or not os.path.exists(a.source):
            os.remove(video)  # free disk; frames are kept
    elif lines:
        dur = lines[-1][0]
    write_outputs(work, title, a.source, a.mode, paras, frames, dur)
    print(json.dumps({"index": os.path.join(work, "index.md"), "frames": len(frames),
                      "paragraphs": len(paras), "duration": ts(dur), "mode": a.mode}))


if __name__ == "__main__":
    main()
