#!/usr/bin/env python3
"""QA for a rendered Short. Exit code 0 only if every check passes.

    python qa.py out/territory_seed90.mp4
    python qa.py out/territory_seed90.mp4 --frames 0,0.5,6,12,18,23.5,24.5,25.5,27,27.97 --frames-dir /tmp/review

Checks: ffprobe conformance (1080x1920, 30 fps, H.264 High yuv420p bt709, AAC stereo, faststart),
length, black frames (blackdetect), loudness and true peak (ebur128), long silences,
motion on the very first frame and how hard the last frame cuts back to the first.
Mix notes (printed as INFO or WARN, they never fail the run, because nobody can listen here): loudness after
a 300 Hz high-pass (what a phone speaker keeps), share of energy below 120 Hz, and sound onsets per 2 s.
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

    # ---- mix notes: warnings only
    notes = []
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", a.video, "-map", "0:a:0",
               "-af", "highpass=f=300,highpass=f=300,ebur128", "-f", "null", "-"]).stderr
    m = re.search(r"I:\s*(-?[\d.]+) LUFS", err[err.rfind("Summary:"):])
    hp = float(m.group(1)) if m else float("nan")
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.video, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(pcm, np.float32).astype(np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    bass = float(spec[np.fft.rfftfreq(len(x), 1 / 16000) < 120].sum() / max(1e-12, spec.sum()))
    nb = len(x) // 800                                            # 50 ms blocks
    db = 10 * np.log10(np.maximum(1e-10, (x[:nb * 800].reshape(nb, 800) ** 2).mean(axis=1)))
    onset = (np.diff(db) > 6.0) & (db[1:] > -40.0)                # level jumps of more than 6 dB in 50 ms
    per2 = [int(onset[i:i + 40].sum()) for i in range(0, len(onset) - 20, 40)]
    lull = best = 0
    for c in per2:
        lull = lull + 1 if c == 0 else 0
        best = max(best, lull)
    notes.append((hp >= loud["lufs"] - 2.0, "loudness above 300 Hz (phone speaker)",
                  f'{hp} LUFS, {hp - loud["lufs"]:+.1f} dB against the full mix (want within 2 dB)'))
    notes.append((bass <= 0.40, "energy below 120 Hz", f"{100 * bass:.0f}% (want 40% or less)"))
    notes.append((best < 3, "sound onsets per 2 s", f"{per2}, longest stretch without one {2 * best} s (want under 6 s)"))

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

    # ---- escalation arc (owner idea 24 Nov): sound and picture should build toward the reveal. Warnings only.
    # The build-up is the first 80% of the Short (the reveal sits in the last ~20%), split into thirds.
    def thirds(arr, frac=0.8):
        n = max(3, int(len(arr) * frac))
        k = n // 3
        return [arr[i * k:(i + 1) * k] for i in range(3)]
    a_lv = [float(np.mean(t)) for t in thirds(db)]                                  # mean level, dB
    a_on = [float(np.mean(t)) * 20 for t in thirds(onset.astype(float))]            # onsets per second
    v_mo = [float(np.mean(t)) for t in thirds(step)]                                # motion
    v_ct = [float(np.mean([f.std() for f in t[::6]])) for t in thirds(g[1:])]      # contrast (luma spread)
    snd_up = a_lv[2] - a_lv[0] >= 1.5 or a_on[2] >= 1.3 * max(0.05, a_on[0])
    pic_up = v_mo[2] >= 1.2 * max(0.05, v_mo[0]) or v_ct[2] >= 1.1 * max(1.0, v_ct[0])
    notes.append((snd_up, "sound builds toward the reveal",
                  f"level by third {[round(v, 1) for v in a_lv]} dB, onsets/s {[round(v, 1) for v in a_on]} "
                  "(want the last third louder by 1.5 dB or 30% busier)"))
    notes.append((pic_up, "picture builds toward the reveal",
                  f"motion by third {[round(v, 2) for v in v_mo]}, contrast {[round(v, 1) for v in v_ct]} "
                  "(want the last third 20% more motion or 10% more contrast)"))

    frames = []
    if a.frames:
        out_dir = a.frames_dir or os.path.splitext(a.video)[0] + "_frames"
        frames = grab_frames(a.video, [float(x) for x in a.frames.split(",")], 30, nframes, out_dir)

    for c in checks:
        print(f'{"PASS" if c["ok"] else "FAIL"}  {c["check"]:<34} {c["detail"]}')
    for ok_, name, detail in notes:
        print(f'{"INFO" if ok_ else "WARN"}  {name:<34} {detail}')
    for p in frames:
        print("frame", p)
    ok = all(c["ok"] for c in checks)
    print(json.dumps(dict(ok=ok, duration=dur, frames=nframes, loudness=loud,
                          video_kbps=round(int(v.get("bit_rate", 0)) / 1000),
                          failed=[c["check"] for c in checks if not c["ok"]],
                          mix_warnings=[n[1] for n in notes if not n[0]])))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
