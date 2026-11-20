"""Audio helpers: numpy-only synth voices and drums, a mixer, reverb, and loudness mastering.

Everything is generated from sine waves and seeded noise. No samples, no soundfonts.
Loudness is measured with ffmpeg's ebur128 filter (the same meter qa.py uses).
"""
import os
import re
import subprocess
import tempfile
import wave

import numpy as np

SR = 48000
TAU = 2 * np.pi


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _perc(t, attack, decay):
    return np.minimum(1.0, t / attack) * np.exp(-t / decay)


# ------------------------------------------------------------------ pitched voices
def _click(n, lo, hi, decay, seed=1):
    """Short band-limited noise burst: the mallet/pick transient that makes a note cut through."""
    m = min(n, int(0.03 * SR))
    out = np.zeros(n)
    out[:m] = bandpass(np.random.default_rng(seed).standard_normal(m), lo, hi) * np.exp(-np.arange(m) / SR / decay)
    return out


def marimba(f, dur=0.8):
    t = _t(dur)
    x = (np.sin(TAU * f * t)
         + 0.34 * np.sin(TAU * 3.93 * f * t) * np.exp(-t / 0.08)
         + 0.16 * np.sin(TAU * 9.5 * f * t) * np.exp(-t / 0.025))
    return x * _perc(t, 0.003, 0.20) + 0.30 * _click(len(t), 2500, 7000, 0.005)


def bell(f, dur=1.6):
    t = _t(dur)
    x = sum(a * np.sin(TAU * r * f * t + p) * np.exp(-t / d)
            for r, a, d, p in ((1.0, 1.0, 0.55, 0), (2.0, 0.35, 0.40, 1), (2.76, 0.30, 0.22, 2),
                               (5.40, 0.14, 0.10, 3), (8.93, 0.06, 0.05, 4)))
    return x * np.minimum(1.0, t / 0.002)


def pluck(f, dur=0.9):
    t = _t(dur)
    x = sum(np.sin(TAU * n * f * t) / n * np.exp(-t * (2.2 + 1.3 * n)) for n in range(1, 13))
    return x * np.minimum(1.0, t / 0.002) + 0.25 * _click(len(t), 2000, 6000, 0.004, 2)


def chip(f, dur=0.5):
    t = _t(dur)
    x = sum(np.sin(TAU * n * f * t) / n * np.exp(-t * 1.1 * n) for n in (1, 3, 5, 7, 9, 11))
    return x * _perc(t, 0.004, 0.18)


def pad(f, dur, attack=0.5, release=0.6):
    t = _t(dur)
    x = sum(np.sin(TAU * f * d * t + p) for d, p in ((0.996, 0.0), (1.0, 1.3), (1.004, 2.1)))
    x = x + 0.25 * np.sin(TAU * 2 * f * t) + 0.08 * np.sin(TAU * 3 * f * t)
    env = np.minimum(1.0, t / attack) * np.minimum(1.0, np.maximum(0.0, (dur - t) / release))
    return x * env / 3.3


def bass(f, dur=0.22):
    t = _t(dur)
    x = sum(a * np.sin(TAU * n * f * t) for n, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.18), (5, 0.1)))
    return x * np.minimum(1.0, t / 0.006) * np.minimum(1.0, np.maximum(0.0, (dur - t) / 0.05))


def glide(f0, f1, dur, decay=None, harm=0.3):
    """Sine that slides from f0 to f1 (exponential), for blips, chirps and risers."""
    t = _t(dur)
    f = f0 * (f1 / f0) ** (t / dur)
    ph = TAU * np.cumsum(f) / SR
    x = np.sin(ph) + harm * np.sin(2 * ph)
    env = np.minimum(1.0, t / 0.004) * np.minimum(1.0, np.maximum(0.0, (dur - t) / 0.02))
    if decay:
        env = env * np.exp(-t / decay)
    return x * env


# ------------------------------------------------------------------ noise and drums
def bandpass(x, lo, hi):
    """Zero-phase FFT band-pass with soft (raised-cosine) edges."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    up = np.clip((f - lo * 0.8) / max(1e-9, lo * 0.2), 0, 1) if lo > 0 else np.ones_like(f)
    down = np.clip((hi * 1.2 - f) / max(1e-9, hi * 0.2), 0, 1)
    g = 0.25 * (1 - np.cos(np.pi * up)) * (1 - np.cos(np.pi * down))
    return np.fft.irfft(X * g, len(x))


def noise(rng, dur, lo, hi):
    return bandpass(rng.standard_normal(int(dur * SR)), lo, hi)


def kick(dur=0.30):
    t = _t(dur)
    f = 50 + 110 * np.exp(-t * 30)
    return (np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t * 14) * np.minimum(1.0, t / 0.002)
            + 0.22 * _click(len(t), 1200, 5000, 0.004, 3))


def hat(rng, dur=0.05):
    t = _t(dur)
    return noise(rng, dur, 7000, 14000) * np.exp(-t * 95)


def shaker(rng, dur=0.08):
    t = _t(dur)
    return noise(rng, dur, 4000, 9000) * np.minimum(1.0, t / 0.012) * np.exp(-t * 45)


def clap(rng, dur=0.16):
    t = _t(dur)
    env = sum(np.exp(-np.maximum(0.0, t - d) * 60) * (t >= d) for d in (0.0, 0.011, 0.023))
    return noise(rng, dur, 900, 3800) * env / 2.2


def tick(f=1750.0, dur=0.06):
    t = _t(dur)
    return (np.sin(TAU * f * t) + 0.5 * np.sin(TAU * f * 1.51 * t)) * _perc(t, 0.001, 0.012)


def boom(rng, dur=1.0, f0=95.0, f1=36.0):
    t = _t(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 9)
    body = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    return (body + 0.5 * noise(rng, dur, 60, 1400) * np.exp(-t * 14)) * np.minimum(1.0, t / 0.003)


def swept_noise(rng, dur, f0, f1, width=0.45):
    """Noise through a band-pass whose centre slides from f0 to f1 (overlap-add STFT)."""
    n = int(dur * SR)
    win, hop = 2048, 512
    x = rng.standard_normal(n + win)
    out = np.zeros(n + win)
    w = np.hanning(win)
    fr = np.fft.rfftfreq(win, 1 / SR)
    for i in range(0, n, hop):
        fc = f0 * (f1 / f0) ** (i / n)
        g = np.exp(-0.5 * (np.log(np.maximum(fr, 1.0) / fc) / width) ** 2)
        out[i:i + win] += np.fft.irfft(np.fft.rfft(x[i:i + win] * w) * g, win) * w
    return out[:n] / 1.5


# ------------------------------------------------------------------ mixing
def place(buf, x, t, gain=1.0, pan=0.0):
    """Add mono x into stereo buf at time t; pan in [-1, 1], constant power."""
    i = int(round(t * SR))
    if i >= len(buf) or i < 0:
        return
    x = x[:len(buf) - i]
    a = (pan + 1) * np.pi / 4
    buf[i:i + len(x), 0] += x * (gain * np.cos(a))
    buf[i:i + len(x), 1] += x * (gain * np.sin(a))


def reverb(x, rng, decay=1.1, wet=0.2, predelay=0.012):
    """Convolution reverb with a synthetic impulse response (decaying filtered noise)."""
    n = int(decay * 3 * SR)
    t = np.arange(n) / SR
    out = np.array(x, dtype=np.float64)
    m = len(x) + n
    size = 1 << (m - 1).bit_length()
    pre = int(predelay * SR)
    for ch in (0, 1):
        ir = bandpass(rng.standard_normal(n), 250, 5500) * np.exp(-t / (decay / 3))
        ir /= np.sqrt(np.sum(ir ** 2))
        y = np.fft.irfft(np.fft.rfft(x[:, ch], size) * np.fft.rfft(ir, size), size)
        out[pre:, ch] += wet * y[:len(x) - pre]
    return out


def eq(x, points):
    """Zero-phase EQ: gain curve in dB through (frequency, dB) points, linear in log-frequency between them."""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    fr, db = zip(*points)
    g = 10 ** (np.interp(np.log(np.maximum(f, 1.0)), np.log(fr), db) / 20)
    return np.stack([np.fft.irfft(np.fft.rfft(x[:, ch]) * g, len(x)) for ch in (0, 1)], axis=1)


def limiter(x, ceiling, look=0.005, hold=0.06):
    """Look-ahead peak limiter. Gain is planned on short blocks, so samples never exceed `ceiling`."""
    blk = 32
    nb = -(-len(x) // blk)
    peak = np.abs(np.pad(x, ((0, nb * blk - len(x)), (0, 0)))).max(axis=1).reshape(nb, blk).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    la, lr = int(look * SR / blk) + 1, int(hold * SR / blk) + 1
    g = np.lib.stride_tricks.sliding_window_view(
        np.pad(need, (lr + 1, la + 1), mode="edge"), la + lr + 3).min(axis=1)
    k = np.hanning(2 * la + 3)[1:-1]            # short smoothing: ~10 ms attack, hold, ~10 ms release
    k /= k.sum()
    g = np.convolve(np.pad(g, (la, la), mode="edge"), k, mode="valid")
    gain = np.interp(np.arange(len(x)), (np.arange(nb) + 0.5) * blk, g)
    return x * gain[:, None]


def write_wav(path, x):
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def loudness(path):
    """Integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) of any media file."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a:0",
                          "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    tail = err[err.rfind("Summary:"):]
    get = lambda pat: float(re.search(pat, tail).group(1))
    return dict(lufs=get(r"I:\s*(-?[\d.]+) LUFS"), lra=get(r"LRA:\s*(-?[\d.]+) LU"),
                true_peak=get(r"Peak:\s*(-?[\d.]+) dBFS"))


def master(x, path, lufs=-14.0, ceiling_db=-2.5):
    """Normalise to the target loudness with the peaks held under ceiling_db, write a 16-bit WAV."""
    ceiling = 10 ** (ceiling_db / 20)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    y = x
    try:
        gain = 0.5 / max(1e-9, np.abs(x).max())
        for _ in range(6):
            y = limiter(x * gain, ceiling)
            write_wav(tmp, y)
            m = loudness(tmp)
            if abs(m["lufs"] - lufs) < 0.1:
                break
            gain *= 10 ** ((lufs - m["lufs"]) / 20)
    finally:
        os.remove(tmp)
    write_wav(path, y)
    return loudness(path)
