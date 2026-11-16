"""Frame pipeline: worker processes draw frames, raw BGRx goes down a pipe into ffmpeg (libx264 + AAC).

A scene is any object with W, H, FPS, nframes and frame(i) -> bytes (cairo FORMAT_RGB24 buffer).
Workers are forked, so the scene (simulation data included) is shared without pickling.
"""
import multiprocessing as mp
import pickle
import subprocess

import cairo

_SCENE = None


def _frame(i):
    return _SCENE.frame(i)


def _init(scene):
    global _SCENE
    _SCENE = scene


def pool(workers, scene=None):
    """Forked pool where available (Linux); elsewhere (Windows) a spawned pool that receives the scene by pickling."""
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork").Pool(workers)
    if scene is None:
        return mp.get_context("spawn").Pool(workers)
    return mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(scene,))


def encode_args(scene, wav_path, out_path, crf=17, preset="medium"):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr0", "-s", f"{scene.W}x{scene.H}", "-r", str(scene.FPS), "-i", "-"]
    if wav_path:
        cmd += ["-i", wav_path]
    cmd += ["-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,"
                   "format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-level", "4.2",
            "-g", str(scene.FPS), "-bf", "2", "-x264-params", "aq-mode=3",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    return cmd + ["-movflags", "+faststart", out_path]


def render(scene, out_path, wav_path=None, workers=4, crf=17, preset="medium"):
    global _SCENE
    _SCENE = scene
    ff = subprocess.Popen(encode_args(scene, wav_path, out_path, crf, preset), stdin=subprocess.PIPE)
    picklable = True
    if "fork" not in mp.get_all_start_methods():
        try:
            pickle.dumps(scene)
        except Exception:
            picklable = False  # scene holds closures; draw frames in this process instead
    if picklable:
        with pool(workers, scene) as p:
            for buf in p.imap(_frame, range(scene.nframes), chunksize=4):
                ff.stdin.write(buf)
    else:
        for i in range(scene.nframes):
            ff.stdin.write(scene.frame(i))
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def still(scene, i, png_path):
    """One frame straight to PNG (no encode), for quick visual checks."""
    buf = bytearray(scene.frame(i))
    cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_RGB24, scene.W, scene.H).write_to_png(png_path)
