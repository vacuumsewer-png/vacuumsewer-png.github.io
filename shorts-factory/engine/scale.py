"""SCALE family core: "How many [objects] until [absurd outcome]?"

One familiar object is added in escalating batches (a visible event every ~0.2-0.55 s, from frame 0), one big
running counter climbs, the camera pulls back past reference objects, the sky goes from cool to warm, and the
outcome happens only when the honest physics says so (the reveal, in the last ~3 s): one flash, a boom, a one-line
"why", then a dissolve back to frame 0 for a clean loop.

An episode subclasses ScaleScene and supplies only what is specific to it:
  * the answer (computed from stated physics constants) and the question / unit / why text;
  * world_height(t): metres above the ground the camera must show at time t;
  * draw_world(ctx, t, s, res): the object, the growing cluster/pile and the reference objects (s = px per metre);
  * optionally event_sound(), reveal_sound(), episode_resources(), sky colors.
World coordinates are metres, x to the right of the anchor, y up from the ground; to_screen() maps them.
On screen there is only the question (top) and the counter (under it); nothing else.
"""
import bisect
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import TAU, Font, clamp, ease_out, ease_out_back, glow, hex_rgb, lerp, mix, ramp

W, H, FPS = 1080, 1920, 30
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
INK = hex_rgb("#070A14")
CREAM = hex_rgb("#FFF8EC")
AMBER = hex_rgb("#FFC23D")
GOLD = hex_rgb("#FFD75A")
HOT = hex_rgb("#FF8A3D")


def schedule(answer, t_rev, dt0=0.45, dt1=0.22):
    """Event times and the count after each event (one every dt0 s at the start, every dt1 s at the end). Starts at 1 on frame 0, +1 at least per event, then
    exponential in time (count = answer ** (t / t_rev)); events get denser toward the reveal; the last event is the
    answer, exactly at t_rev. No earlier count reaches the answer."""
    ts = [0.0]
    t = 0.0
    while True:
        t += dt0 + (dt1 - dt0) * (t / t_rev)
        if t > t_rev - 0.6 * dt1:
            break
        ts.append(t)
    cs = [1]
    for t in ts[1:]:
        cs.append(min(answer - 1, max(cs[-1] + 1, int(round(answer ** (t / t_rev))))))
    ts.append(t_rev)
    cs.append(answer)
    return ts, cs


class ScaleScene:
    W, H, FPS = W, H, FPS
    T_TOTAL = 25.0
    T_REV = 22.0
    XFADE = (0.42, 0.06)          # dissolve to frame 0 between T_TOTAL - 0.42 and T_TOTAL - 0.06
    ROOT = 45
    GY = 1480.0                   # ground line on screen
    AX = 480.0                    # screen x of world x = 0
    HX = 475.0                    # text center (keeps clear of the right 12%)
    TOP_PX = 760.0                # the camera keeps the world top below this screen y
    S0, DRIFT = 200.0, 0.012      # opening px per metre, slow pull-back from frame 0
    QUESTION = ("HOW MANY", "ITEMS?")
    UNIT = ("ITEM", "ITEMS")
    WHY = ("", "")
    SKY_COOL = (hex_rgb("#5FA8E8"), hex_rgb("#CFEAFB"))
    SKY_WARM = (hex_rgb("#E7744A"), hex_rgb("#FFD49A"))
    EQ = ((50, -6.0), (120, -2.5), (300, -0.5), (900, 0.0), (3000, 1.5), (8000, 1.0), (16000, -4.0))
    DRIVE = 3.0
    CEILING_DB = -3.4

    def __init__(self, seed, answer):
        self.seed = seed
        self.answer = int(answer)
        self.ev_t, self.ev_c = schedule(self.answer, self.T_REV)
        self.nframes = int(round(self.T_TOTAL * FPS))
        self.dur = self.nframes / FPS
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    # ------------------------------------------------------------ hooks for the episode
    def world_height(self, t):
        raise NotImplementedError

    def draw_world(self, ctx, t, s, res):
        raise NotImplementedError

    def episode_resources(self, res):
        pass

    # ------------------------------------------------------------ time helpers
    def progress(self, t):
        return clamp(t / self.T_REV)

    def event_index(self, t):
        return bisect.bisect_right(self.ev_t, t) - 1

    def count_f(self, t):
        """Displayed count as a float: each event rolls the counter up over a short time."""
        e = self.event_index(t)
        if e <= 0:
            return 1.0
        t0 = self.ev_t[e]
        roll = 0.0 if e == len(self.ev_t) - 1 else min(0.16, 0.7 * (t0 - self.ev_t[e - 1]))
        if roll <= 0 or t - t0 >= roll:
            return float(self.ev_c[e])
        return lerp(self.ev_c[e - 1], self.ev_c[e], ease_out((t - t0) / roll, 2))

    def count(self, t):
        return int(round(self.count_f(t)))

    def since_event(self, t):
        e = self.event_index(t)
        return t - self.ev_t[max(0, e)], e

    INTRO_ZOOM, INTRO_T = 0.16, 0.9   # the camera is already pulling back on frame 0 (motion from the first frame)

    def scale(self, t):
        s = min(self.S0 * math.exp(-self.DRIFT * t), (self.GY - self.TOP_PX) / self.world_height(t))
        return s * (1.0 + self.INTRO_ZOOM * (1.0 - ease_out(clamp(t / self.INTRO_T), 2)))

    def to_screen(self, x, y, s):
        return self.AX + x * s, self.GY - y * s

    def world_ctx(self, ctx, s):
        """Switch ctx to world metres (y up). Caller wraps in save/restore."""
        ctx.translate(self.AX, self.GY)
        ctx.scale(s, -s)

    # ------------------------------------------------------------ frames
    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), font=font)
            self.episode_resources(self._res)
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        t = f / FPS
        a = ramp(t, self.T_TOTAL - self.XFADE[0], self.T_TOTAL - self.XFADE[1])
        if a <= 0.0:
            self.draw(ctx, t, res)
        else:
            ctx.push_group()
            self.draw(ctx, t, res)
            ctx.pop_group_to_source()
            ctx.paint()
            ctx.push_group()
            self.draw(ctx, 0.0, res)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(a)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    def draw(self, ctx, t, res):
        s = self.scale(t)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.draw_sky(ctx, t, s)
        self.draw_ground(ctx, t, s)
        self.draw_world(ctx, t, s, res)
        d = t - self.T_REV
        if d >= 0:
            self.draw_flash(ctx, t, s, d)
        self.draw_question(ctx, res["font"], t, d)
        self.draw_counter(ctx, res["font"], t, d)

    # ------------------------------------------------------------ backdrop (cool -> warm)
    def warmth(self, t):
        return ease_out(clamp(t / self.T_REV), 1.6) ** 1.3

    def draw_sky(self, ctx, t, s):
        k = self.warmth(t)
        top = mix(self.SKY_COOL[0], self.SKY_WARM[0], k)
        hor = mix(self.SKY_COOL[1], self.SKY_WARM[1], k)
        g = cairo.LinearGradient(0, 0, 0, self.GY)
        g.add_color_stop_rgb(0, *top)
        g.add_color_stop_rgb(1, *hor)
        ctx.set_source(g)
        ctx.rectangle(0, 0, W, self.GY + 1)
        ctx.fill()
        # low sun: appears and warms with the build
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, 930, self.GY - 520 + 300 * k, 700, mix((0.9, 0.95, 1.0), (1.0, 0.72, 0.35), k), 0.18 + 0.30 * k)
        ctx.restore()
        # soft clouds drifting, a little parallax with the zoom
        z = (s / self.S0) ** 0.25
        for i in range(6):
            h1 = (i * 0.618034) % 1.0
            cx = (h1 * 1500 + 14 * t * (0.6 + 0.4 * h1)) % 1500 - 210
            cy = 760 + 520 * ((i * 0.381966 + 0.2) % 1.0)
            r = (110 + 70 * h1) * z
            ctx.save()
            ctx.translate(cx, cy)
            ctx.scale(1.0, 0.42)
            for j, (ox, rr) in enumerate(((-0.8, 0.7), (0.0, 1.0), (0.85, 0.75))):
                gg = cairo.RadialGradient(ox * r, 0, 0, ox * r, 0, rr * r)
                gg.add_color_stop_rgba(0, 1, 1, 1, 0.32)
                gg.add_color_stop_rgba(1, 1, 1, 1, 0.0)
                ctx.set_source(gg)
                ctx.arc(ox * r, 0, rr * r, 0, TAU)
                ctx.fill()
            ctx.restore()

    def draw_ground(self, ctx, t, s):
        k = self.warmth(t)
        gy = self.GY
        g = cairo.LinearGradient(0, gy, 0, H)
        g.add_color_stop_rgb(0, *mix((0.36, 0.62, 0.30), (0.52, 0.55, 0.26), k))
        g.add_color_stop_rgb(1, *mix((0.12, 0.26, 0.12), (0.22, 0.20, 0.10), k))
        ctx.set_source(g)
        ctx.rectangle(0, gy, W, H - gy)
        ctx.fill()
        # road the object stands on: its depth on screen follows the zoom
        rd = max(10.0, 1.6 * s)
        g = cairo.LinearGradient(0, gy, 0, gy + rd)
        g.add_color_stop_rgb(0, 0.30, 0.31, 0.34)
        g.add_color_stop_rgb(1, 0.20, 0.21, 0.24)
        ctx.set_source(g)
        ctx.rectangle(0, gy, W, rd)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.55)
        dash = 3.0 * s
        x0 = self.AX - (self.AX // (2 * dash) + 1) * 2 * dash
        x = x0
        while x < W:
            ctx.rectangle(x, gy + rd * 0.48, dash, max(2.0, 0.12 * s))
            x += 2 * dash
        ctx.fill()
        ctx.set_source_rgba(0, 0, 0, 0.25)
        ctx.rectangle(0, gy, W, 2)
        ctx.fill()

    def draw_flash(self, ctx, t, s, d):
        a = math.exp(-d / 0.12)
        if a < 0.01:
            return
        ctx.save()
        ctx.set_operator(cairo.OPERATOR_ADD)
        x, y = self.to_screen(0.0, 2.0, s)
        glow(ctx, x, y, 1100, (1.0, 0.95, 0.82), 0.85 * a)
        ctx.set_source_rgba(1.0, 0.97, 0.90, 0.30 * math.exp(-d / 0.06))
        ctx.paint()
        ctx.restore()

    # ------------------------------------------------------------ the only text: question (then why) and counter
    def _fit(self, font, lines, maxw, cap):
        return min(cap, *(maxw / font.width(x, 1.0, 0.0) for x in lines if x))

    def draw_question(self, ctx, font, t, d):
        aq = 1.0 - ramp(d, 0.30, 0.50) if d > 0 else 1.0
        if aq > 0.01:
            sz = self._fit(font, self.QUESTION, 880.0, 96.0)
            font.draw(ctx, self.QUESTION[0], self.HX, 250, sz, CREAM, anchor="c", alpha=aq, stroke=INK, stroke_w=14,
                      shadow=7)
            font.draw(ctx, self.QUESTION[1], self.HX, 250 + sz * 1.08, sz, AMBER, anchor="c", alpha=aq, stroke=INK,
                      stroke_w=14, shadow=7)
        if d > 0.45 and self.WHY[0]:
            a = ramp(d, 0.45, 0.65)
            sz = self._fit(font, self.WHY, 930.0, 76.0)
            k = 0.9 + 0.1 * ease_out_back(ramp(d, 0.45, 0.75))
            ctx.save()
            ctx.translate(self.HX, 250)
            ctx.scale(k, k)
            font.draw(ctx, self.WHY[0], 0, 0, sz, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=12, shadow=6)
            font.draw(ctx, self.WHY[1], 0, sz * 1.15, sz, GOLD, anchor="c", alpha=a, stroke=INK, stroke_w=12, shadow=6)
            ctx.restore()

    def draw_counter(self, ctx, font, t, d):
        n = self.count(t)
        k = self.progress(t)
        age, e = self.since_event(t)
        pop = 1.0
        if e > 0 and d < 0:
            pop += (0.06 + 0.08 * k) * math.exp(-age / 0.07)
        col = mix(CREAM, AMBER, ease_out(k, 2))
        if d >= 0:
            pop = 1.0 + 0.22 * ease_out_back(clamp(d / 0.35)) - 0.06 * ramp(d, 0.35, 0.8)
            col = GOLD
        txt = f"{n:,}"
        sz = 168.0
        y = 610.0
        ctx.save()
        ctx.translate(self.HX, y - 60)
        ctx.scale(pop, pop)
        if d >= 0:
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_ADD)
            glow(ctx, 0, 0, 330, (1.0, 0.75, 0.3), 0.5 * (0.5 + 0.5 * math.exp(-d / 0.4)))
            ctx.restore()
        font.draw(ctx, txt, 0, 60, sz, col, anchor="c", stroke=INK, stroke_w=16, shadow=8)
        unit = self.UNIT[0] if n == 1 else self.UNIT[1]
        font.draw(ctx, unit, 0, 60 + 74, 56, CREAM, anchor="c", stroke=INK, stroke_w=10, tracking=3.0)
        ctx.restore()

    # ------------------------------------------------------------ sound
    def event_sound(self, i, t, k, rng):
        """Sound for event i at time t (k = build progress 0..1). Returns a list of (mono, gain, pan, dt)."""
        f0 = 380.0 * 2 ** (1.6 * k)
        return [(au.glide(f0, f0 * 1.5, 0.10, decay=0.05, harm=0.4), 0.20, 0.0, 0.0)]

    def reveal_sound(self, fx, rng):
        pass

    def audio(self):
        rng = np.random.default_rng(self.seed + 777)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        bed[:n] = au.tension_bed(self.dur, self.T_REV, root=self.ROOT, seed=self.seed % 97 + 3, gain=0.55)
        for i, te in enumerate(self.ev_t[1:-1], start=1):
            if te > self.T_REV - 0.35:                      # keep the short hush before the reveal
                continue
            for x, g, pan, dt in self.event_sound(i, te, self.progress(te), rng):
                au.place(fx, x, te + dt, g, pan)
        self.reveal_sound(fx, rng)
        for j, m_ in enumerate((60, 64, 67, 72)):          # rising arpeggio under the dissolve back to frame 0
            au.place(fx, au.marimba(au.hz(m_ + self.ROOT - 45), 0.4), self.T_TOTAL - 0.62 + 0.13 * j, 0.12)
        mix_ = au.reverb(fx, rng, decay=1.0, wet=0.12) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        out = au.eq(out, self.EQ)
        out[:int(0.04 * sr)] *= np.linspace(0, 1, int(0.04 * sr))[:, None]
        out[-int(0.25 * sr):] *= np.linspace(1, 0, int(0.25 * sr))[:, None]
        pk = np.abs(out).max()
        return np.tanh(out * (self.DRIVE / max(1e-9, pk)))

    # ------------------------------------------------------------ metadata
    def base_drama(self, **extra):
        pre = [c for t, c in zip(self.ev_t, self.ev_c) if t < self.T_REV]
        d = dict(seed=self.seed, score=1.0, valid=True, winner=f"{self.answer:,}", answer=self.answer,
                 events=len(self.ev_t), last_count_before_reveal=pre[-1], reveal_s=self.T_REV,
                 first_counts=self.ev_c[:8])
        d.update(extra)
        return d
