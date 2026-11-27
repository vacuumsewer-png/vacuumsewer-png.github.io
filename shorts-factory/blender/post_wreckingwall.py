"""Post for the Wrecking Wall 3D Short: overlay (Anton), sound from the real hit/landing frames, encode, metadata.

    py blender/post_wreckingwall.py out/3d_wreckingwall

Reads <dir>/f_0001.png ... and <dir>/sim.json + sim.npz written by blender/wreckingwall3d.py.
Writes out/wreckingwall_3d.mp4 and out/wreckingwall_3d.json. On screen: one question line and one hit
counter; the measured answer and a one-line physics note appear only in the last 3 s.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from engine import audio  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "3d_wreckingwall")
NAME = "wreckingwall"
FPS, W, H = 30, 1080, 1920
REVEAL_T = 22.0
LOOP_FRAMES = 12
FONT = os.path.join(ROOT, "fonts", "Anton-Regular.ttf")
QUESTION = ["HOW MANY BRICKS", "GET KNOCKED OUT?"]
CYAN = (60, 225, 255)

sim = json.load(open(os.path.join(SRC, "sim.json")))
npz = np.load(os.path.join(SRC, "sim.npz"))
pos = npz["pos"]
N = pos.shape[0]
DUR = N / FPS
ANSWER = sim["knocked_out_by_frame"][int(REVEAL_T * FPS) - 1]          # the count at the moment it is shown
HIT_F = sim["hit_frames"]
SPEEDS = sim["hit_speed_kmh"]


# ------------------------------------------------------------------ sound
def build_audio(path):
    rng = np.random.default_rng(27)
    n = int(DUR * audio.SR)
    mix = audio.tension_bed(DUR, REVEAL_T, root=43, seed=11)
    vmax = max(SPEEDS)
    for hf, v in zip(HIT_F, SPEEDS):
        t = (hf - 1) / FPS
        k = v / vmax
        audio.place(mix, audio.boom(rng, 1.1, f0=120, f1=42), t, 0.35 + 0.55 * k)
        crunch = audio.noise(rng, 0.6, 500, 7000) * np.exp(-np.arange(int(0.6 * audio.SR)) / audio.SR / (0.05 + 0.12 * k))
        audio.place(mix, crunch, t, 0.25 + 0.45 * k)
    # bricks landing: a falling brick whose vertical speed suddenly stops
    z = pos[:, :, 2]
    vz = np.diff(z, axis=0) * FPS
    landings = 0
    for f in range(1, vz.shape[0]):
        hit = np.where((vz[f - 1] < -1.8) & (vz[f] > -0.6) & (z[f + 1] < 1.2))[0]
        if not len(hit):
            continue
        landings += len(hit)
        speed = float(np.mean(-vz[f - 1][hit]))
        x = float(np.mean(pos[f + 1, hit, 0]))
        dur = 0.09
        clack = audio.bandpass(rng.standard_normal(int(dur * audio.SR)), 250, 3200)
        clack *= np.exp(-np.arange(len(clack)) / audio.SR / 0.025)
        gain = min(0.55, 0.10 * np.sqrt(len(hit)) * min(2.0, speed / 3.0))
        audio.place(mix, clack, (f + 1) / FPS, gain, pan=max(-0.8, min(0.8, x / 3.0)))
        audio.place(mix, audio.kick(0.12), (f + 1) / FPS, gain * 0.5)
    mix = mix[:n]
    loud = audio.master(mix, path)
    return loud, landings


# ------------------------------------------------------------------ overlay
def font(size):
    return ImageFont.truetype(FONT, size)


F_Q, F_C, F_A, F_A2, F_NOTE = font(96), font(84), font(250), font(80), font(50)


def text_c(d, y, s, f, fill=(255, 255, 255), stroke=9):
    w = d.textlength(s, font=f)
    d.text(((W - w) / 2, y), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))


def hits_done(f):
    return sum(1 for hf in HIT_F if f >= hf)


def overlay(img, f):
    d = ImageDraw.Draw(img)
    t = (f - 1) / FPS
    if t < REVEAL_T:
        y0 = 430
        for i, line in enumerate(QUESTION):
            text_c(d, y0 + i * 118, line, F_Q)
        h = hits_done(f)
        if h:
            text_c(d, y0 + 270, f"HIT {h} / {len(HIT_F)}", F_C, fill=CYAN, stroke=8)
    else:
        k = min(1.0, (t - REVEAL_T) / 0.2)
        size = int(250 * (0.6 + 0.4 * k)) if k < 1 else 250
        fa = F_A if size == 250 else font(size)
        text_c(d, 380 + (250 - size) // 2, f"{ANSWER}", fa, fill=CYAN, stroke=12)
        text_c(d, 670, "BRICKS KNOCKED OUT", F_A2)
        if t >= REVEAL_T + 0.6:
            note = f"Higher swing = faster ball: {SPEEDS[0]:.0f} to {SPEEDS[-1]:.0f} km/h"
            text_c(d, 790, note, F_NOTE, stroke=6)
    return img


def frame(f):
    return Image.open(os.path.join(SRC, f"f_{f:04d}.png")).convert("RGB")


def encode(wav, mp4):
    vf = ("scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,"
          "format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a", "-vf", vf,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high", "-level", "4.2",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    first = overlay(frame(1), 1)
    for f in range(1, N + 1):
        img = first if f == 1 else overlay(frame(f), f)
        left = N - f
        if left < LOOP_FRAMES:                       # dissolve back to frame 1 so the loop cut is seamless
            a = 1.0 - left / LOOP_FRAMES
            img = Image.blend(img, first, a)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    out = os.path.join(ROOT, "out")
    wav = os.path.join(out, f"{NAME}_3d.wav")
    mp4 = os.path.join(out, f"{NAME}_3d.mp4")
    t0 = time.time()
    loud, landings = build_audio(wav)
    t1 = time.time()
    encode(wav, mp4)
    os.remove(wav)
    t2 = time.time()
    timing = open(os.path.join(SRC, "timing.txt")).read().strip()
    meta = dict(
        title="4 Swings, Each One Higher… How Many Bricks Come Out?",
        description=("Comment your number 👇\n"
                     f"A steel wrecking ball on a {sim['pendulum_m']:.1f} m cable swings into a wall of "
                     f"{sim['bricks']} bricks four times, released from {', '.join(str(a) for a in sim['release_angle_deg'])} "
                     "degrees. A higher release means a faster ball: "
                     f"{', '.join(f'{v:.0f}' for v in SPEEDS)} km/h at impact.\n"
                     "Every brick is a real rigid body in a Blender physics simulation, and the count is measured, "
                     "not picked: a brick counts once it ends up more than 30 cm from where it was laid.\n"
                     "Made by Predixel: I set the swings, the simulation decides the rest. No footage, samples or stock media.\n"
                     "#simulation #physics #satisfying"),
        tags=["simulation", "wrecking ball", "brick wall", "physics", "blender", "3d", "how many",
              "satisfying", "destruction", "crush", "made with code", "oddly satisfying"],
        pinned_comment="Comment your number 👇 How many of the 172 bricks did the wrecking ball knock out?",
        made_for_kids=False,
        synthetic_media_disclosure=False,
        synthetic_media_disclosure_reason="Clearly animated 3D simulation of a wrecking ball and toy-like brick wall; "
                                          "not realistic footage of people, places or events.",
        family="CRUSH",
        format="3D (Blender 4.5 EEVEE)",
        duration_seconds=round(DUR, 3),
        answer=ANSWER,
        drama=dict(answer_bricks=ANSWER, bricks_total=sim["bricks"], hit_seconds=sim["hit_seconds"],
                   hit_speed_kmh=SPEEDS, release_angle_deg=sim["release_angle_deg"],
                   knocked_out_after_hit=sim["knocked_out_after_hit"], final_count=sim["answer"],
                   reveal_s=REVEAL_T, landing_sounds=landings, rule=sim["rule"]),
        on_screen=dict(question=" ".join(QUESTION), counter="HIT n / 4",
                       physics_note=f"Higher swing = faster ball: {SPEEDS[0]:.0f} to {SPEEDS[-1]:.0f} km/h"),
        render=dict(blender=timing, post_audio_seconds=round(t1 - t0, 1), post_video_seconds=round(t2 - t1, 1),
                    loudness=loud),
    )
    with open(os.path.join(out, f"{NAME}_3d.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(json.dumps(meta["drama"]))
    print(mp4)


if __name__ == "__main__":
    main()
