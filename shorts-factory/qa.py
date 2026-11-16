#!/usr/bin/env python3
"""QA for a rendered Short. Exit code 0 only if every check passes.

    python qa.py out/territory_seed90.mp4
    python qa.py out/territory_seed90.mp4 --frames 0,0.5,6,12,18,23.5,24.5,25.5,27,27.97 --frames-dir /tmp/review

Checks: ffprobe conformance (1080x1920, 30 fps, H.264 High yuv420p bt709, AAC stereo, faststart),
length, black frames (blackdetect), loudness and true peak (ebur128), long silences,
motion on the very first frame and how hard the last frame cuts back to the first.
--frames grabs PNG stills at the given timestamps (seconds) for a human or model to look at.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

from engine.audio import loudness

SPEC = dict(width=1080, height=1920, fps="30/1", min_s=24.0, max_s=32.0, lufs=-14.0, lufs_tol=1.0, max_tp=-1.0)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def top_level_boxes(path):
    out = []
    with open(path, "rb") as fh:
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head)
            if size == 1:
                size = struct.unpack(">Q", fh.read(8))[0]
                fh.seek(size - 16, 1)
            elif size == 0:
                out.append(kind.decode("latin1"))
                break
            else:
                fh.seek(size - 8, 1)
            out.append(kind.decode("latin1"))
    return out


def grab_frames(path, times, fps, nframes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    idx = sorted({min(nframes - 1, max(0, round(t * fps))) for t in times})
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    tmp = os.path.join(out_dir, "grab_%03d.png")
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", f"select='{sel}'", "-vsync", "0", tmp])
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    paths = []
    for k, i in enumerate(idx):
        dst = os.path.join(out_dir, f"frame_{i:04d}_{i / fps:06.3f}s.png")
        os.replace(tmp % (k + 1), dst)
        paths.append(dst)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frames", help="comma-separated timestamps in seconds to extract as PNG")
    ap.add_argument("--frames-dir", default=None, help="where to put the PNGs (default: <video>_frames)")
    a = ap.parse_args()
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, ok=bool(ok), detail=str(detail)))

    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", a.video]).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    au = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"]["duration"])
    nframes = int(v["nb_frames"])
    check("video codec H.264 High", v["codec_name"] == "h264" and v.get("profile") == "High",
          f'{v["codec_name"]} {v.get("profile")}')
    check("pixel format yuv420p", v["pix_fmt"] == "yuv420p", v["pix_fmt"])
    check("resolution 1080x1920", (v["width"], v["height"]) == (SPEC["width"], SPEC["height"]),
          f'{v["width"]}x{v["height"]}')
    check("frame rate 30 fps", v["r_frame_rate"] == SPEC["fps"] and v["avg_frame_rate"] == SPEC["fps"],
          f'r={v["r_frame_rate"]} avg={v["avg_frame_rate"]}')
    colour = (v.get("color_space"), v.get("color_primaries"), v.get("color_transfer"), v.get("color_range"))
    check("colour bt709 limited range", colour == ("bt709", "bt709", "bt709", "tv"), colour)
    check("audio AAC stereo 48 kHz", au is not None and au["codec_name"] == "aac" and au["channels"] == 2
          and au["sample_rate"] == "48000", au and f'{au["codec_name"]} {au["channels"]}ch {au["sample_rate"]} Hz')
    abr = int(au.get("bit_rate", 0)) if au else 0
    check("audio bitrate about 192k", 160_000 <= abr <= 215_000, f"{abr} b/s")
    check("length 24-32 s", SPEC["min_s"] <= dur <= SPEC["max_s"], f"{dur:.3f} s, {nframes} frames")
    boxes = top_level_boxes(a.video)
    check("faststart (moov before mdat)", "moov" in boxes and boxes.index("moov") < boxes.index("mdat"), boxes)

    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", a.video, "-vf", "blackdetect=d=0.1:pic_th=0.98:pix_th=0.10",
               "-an", "-f", "null", "-"]).stderr
    black = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)
    check("no black frames", not black, black or "none")

    loud = loudness(a.video)
    check("loudness about -14 LUFS", abs(loud["lufs"] - SPEC["lufs"]) <= SPEC["lufs_tol"], f'{loud["lufs"]} LUFS')
    check("true peak under -1 dBTP", loud["true_peak"] < SPEC["max_tp"], f'{loud["true_peak"]} dBTP')
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", a.video, "-af", "silencedetect=noise=-50dB:d=0.6",
               "-vn", "-f", "null", "-"]).stderr
    silent = re.findall(r"silence_start: ([\d.]+)", err)
    check("no silence longer than 0.6 s", not silent, silent or "none")

    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.video, "-vf", "scale=270:480,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    g = np.frombuffer(raw, np.uint8).reshape(-1, 480, 270).astype(np.int16)
    step = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))          # mean abs luma change between frames
    typical = float(np.median(step))
    first, loop = float(step[0]), float(np.abs(g[0] - g[-1]).mean())
    check("motion on the first frame", first >= 0.5 * typical, f"frame0->1 change {first:.2f} (typical {typical:.2f})")
    check("last frame cuts cleanly to first", loop <= 4 * typical,
          f"last->first change {loop:.2f} (typical {typical:.2f}, largest in-video {step.max():.2f})")
    check("first frame is not dark or empty", g[0].mean() > 12 and g[0].std() > 12,
          f"luma mean {g[0].mean():.1f}, std {g[0].std():.1f}")

    frames = []
    if a.frames:
        out_dir = a.frames_dir or os.path.splitext(a.video)[0] + "_frames"
        frames = grab_frames(a.video, [float(x) for x in a.frames.split(",")], 30, nframes, out_dir)

    for c in checks:
        print(f'{"PASS" if c["ok"] else "FAIL"}  {c["check"]:<34} {c["detail"]}')
    for p in frames:
        print("frame", p)
    ok = all(c["ok"] for c in checks)
    print(json.dumps(dict(ok=ok, duration=dur, frames=nframes, loudness=loud,
                          video_kbps=round(int(v.get("bit_rate", 0)) / 1000),
                          failed=[c["check"] for c in checks if not c["ok"]])))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
