#!/usr/bin/env python3
"""Shorts factory CLI.

    python make.py territory --out out/              score --seeds seeds for drama, render the best one
    python make.py territory --seed 123 --out out/   render one specific seed
    python make.py territory --seed 123 --stills 0,0.5,12 --out /tmp/stills   PNG stills only (fast look)
"""
import argparse
import importlib
import json
import multiprocessing as mp
import os
import time

from engine import audio, pipeline

SCENES = sorted(f[:-3] for f in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
                if f.endswith(".py") and not f.startswith("_"))


def search(scene_mod, n, workers):
    """Simulate seeds 1..n without rendering and rank them by the scene's drama score."""
    with pipeline.pool(workers) as pool:
        res = pool.map(scene_mod.score_seed, range(1, n + 1), chunksize=8)
    return sorted(res, key=lambda d: (-d["score"], d["seed"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", choices=SCENES)
    ap.add_argument("--seed", type=int, help="render this seed (default: search for the most dramatic one)")
    ap.add_argument("--seeds", type=int, default=600, help="seeds to score when --seed is not given")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--stills", help="comma-separated times in seconds: write PNG stills instead of a video")
    a = ap.parse_args()

    mod = importlib.import_module(f"scenes.{a.scene}")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    seed, searched = a.seed, None
    if seed is None:
        ranked = search(mod, a.seeds, a.workers)
        seed = ranked[0]["seed"]
        searched = dict(seeds_scored=a.seeds, seconds=round(time.time() - t0, 1),
                        top5=[dict(seed=d["seed"], score=d["score"]) for d in ranked[:5]])
        print(f"seed search: {a.seeds} seeds in {searched['seconds']} s -> seed {seed} "
              f"(score {ranked[0]['score']}; next best {[d['seed'] for d in ranked[1:5]]})")
    scene = mod.Scene(seed)
    base = os.path.join(a.out, f"{a.scene}_seed{seed}")

    if a.stills:
        for s in a.stills.split(","):
            f = min(scene.nframes - 1, round(float(s) * scene.FPS))
            pipeline.still(scene, f, f"{base}_{f:04d}.png")
            print(f"{base}_{f:04d}.png")
        return

    t1 = time.time()
    loud = audio.master(scene.audio(), base + ".wav")
    t2 = time.time()
    pipeline.render(scene, base + ".mp4", base + ".wav", a.workers)
    os.remove(base + ".wav")
    t3 = time.time()
    meta = scene.metadata()
    meta["seed_search"] = searched
    meta["render"] = dict(frames=scene.nframes, fps=scene.FPS, duration_seconds=scene.nframes / scene.FPS,
                          audio_seconds=round(t2 - t1, 1), video_seconds=round(t3 - t2, 1),
                          total_seconds=round(t3 - t0, 1), premaster_loudness=loud)
    with open(base + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta["drama"], indent=2))
    print(f"audio {t2 - t1:.1f} s, video {t3 - t2:.1f} s ({scene.nframes / (t3 - t2):.1f} fps), total {t3 - t0:.1f} s")
    print(base + ".mp4")
    print(base + ".json")


if __name__ == "__main__":
    main()
