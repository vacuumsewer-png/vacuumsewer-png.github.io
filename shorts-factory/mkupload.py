"""Make a sub-10 MB upload copy (2-pass x264, audio stream copied) in out/upload/.

    py mkupload.py out/<file>.mp4 [target_mb]

The Chrome upload tool caps files at 10 MB. Audio is copied, not re-encoded, so loudness and
true peak stay exactly as mastered. Run qa.py on the result.
"""
import json
import os
import subprocess
import sys


def probe(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,bit_rate",
         "-of", "json", path])
    j = json.loads(out)
    dur = float(j["format"]["duration"])
    abr = 0
    for s in j["streams"]:
        if s["codec_type"] == "audio":
            abr = int(s.get("bit_rate") or 192000)
    return dur, abr


def main():
    src = sys.argv[1]
    target_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 9.3
    dur, abr = probe(src)
    dst_dir = os.path.join(os.path.dirname(src), "upload")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    vbr = int((target_mb * 1024 * 1024 * 8 / dur - abr) * 0.985)
    log = os.path.join(dst_dir, "_2pass_" + os.path.splitext(os.path.basename(src))[0])
    common = ["-c:v", "libx264", "-preset", "slow", "-b:v", str(vbr), "-profile:v", "high",
              "-level", "4.2", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
              "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
              "-passlogfile", log]
    null = "NUL" if os.name == "nt" else "/dev/null"
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", src] + common +
                          ["-pass", "1", "-an", "-f", "mp4", null])
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", src] + common +
                          ["-pass", "2", "-c:a", "copy", "-movflags", "+faststart", dst])
    for f in os.listdir(dst_dir):
        if f.startswith("_2pass_"):
            os.remove(os.path.join(dst_dir, f))
    size = os.path.getsize(dst) / 1024 / 1024
    print(f"{dst}: {size:.2f} MB, video {vbr // 1000} kbps, {dur:.2f} s")
    if size >= 10:
        sys.exit("too big: rerun with a smaller target")


if __name__ == "__main__":
    main()
