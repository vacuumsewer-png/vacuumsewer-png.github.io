"""One chicken: a 24 s captioned cartoon parable. Three people get the same chicken and choose differently.

Slice 1 (rough): story format, not a simulation, so the seed only exists to satisfy make.py.
Shortcuts to clean up later: no walking or mouth animation, flat panels instead of a drawn town,
placeholder music bed, never heard by anyone.
"""
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import Font, clamp, ease_in_out, ease_out, ease_out_back, hex_rgb, lerp, ramp, rounded_rect

W, H, FPS = 1080, 1920, 30
DUR = 24.0
NF = int(DUR * FPS)
TAU = math.tau

INK = hex_rgb("#2A1B14")
SKY0, SKY1 = hex_rgb("#FFE2A8"), hex_rgb("#FFC27A")
PANEL = [hex_rgb(c) for c in ("#F7B7A3", "#BFE3C0", "#C9C3F2")]
BODY = [hex_rgb(c) for c in ("#E4572E", "#2E9E6B", "#6C5CE7")]
CREAM, WHITE = hex_rgb("#FFF6E0"), (1.0, 1.0, 1.0)
RED, ORANGE, YELLOW, BROWN = hex_rgb("#E23B3B"), hex_rgb("#F59E1B"), hex_rgb("#FFD84A"), hex_rgb("#A65A2A")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")

LX, LW, LH = 50, 900, 340                      # lanes stay inside the safe area (right 12%, bottom 20%)
LTOP = (430, 790, 1150)
CY = [y + LH / 2 - 20 for y in LTOP]
T_NIGHT, T_WEEKS, T_YEAR, T_HIRE, T_END, T_RESET = 3.0, 7.5, 12.0, 17.0, 21.0, 23.4

HEADS = [(0.0, "EVERYONE GETS 1 CHICKEN", "WHO ENDS UP RICH?"),
         (T_NIGHT, "NIGHT ONE", "3 PEOPLE, 3 CHOICES"),
         (T_WEEKS, "3 WEEKS LATER", ""),
         (T_YEAR, "1 YEAR LATER", ""),
         (T_HIRE, "TOO MANY TO MIND", "SO THE OWNER HIRES HELP"),
         (T_END, "SAME CHICKEN", "3 DIFFERENT ENDINGS"),
         (T_RESET + 0.1, "EVERYONE GETS 1 CHICKEN", "WHO ENDS UP RICH?")]
LABELS = [[(3.9, "ATE IT THAT NIGHT"), (7.9, "NOTHING LEFT"), (12.3, "STILL NOTHING"), (18.5, "WORKS FOR EGGS"),
           (T_END + 0.2, "POOR")],
          [(4.5, "KEPT THE HEN"), (8.1, "AN EGG A DAY"), (12.5, "ENOUGH TO LIVE ON"), (T_END + 0.5, "GETTING BY")],
          [(5.6, "FOUND A ROOSTER"), (8.4, "CHICKS HATCH"), (12.8, "A WHOLE FLOCK"), (17.4, "PAYS IN EGGS"),
           (T_END + 0.8, "RICH")]]
FLOCK = [(330 + 84 * (k % 7) + (20 if k // 7 else 0), -62 + 112 * (k // 7), 12.2 + 0.24 * k) for k in range(14)]


def pop(t, t0, d=0.35):
    return ease_out_back(ramp(t, t0, t0 + d))


def ink(ctx, fill, lw=7.0, a=1.0):
    ctx.set_source_rgba(*fill, a)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, a)
    ctx.set_line_width(lw)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()


def ellipse(ctx, x, y, rx, ry):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(rx, ry)
    ctx.new_path()
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()


def at(ctx, x, y, s, rot=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(rot)
    ctx.scale(s, s)


def person(ctx, x, y, i, mood, t):
    """Capsule villager. i picks colour and headwear (tuft, bandana, glasses) so colour is not the only cue."""
    at(ctx, x, y + 4 * math.sin(t * 2.2 + i), 1.0)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(9)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for dx in (-22, 22):
        ctx.move_to(dx, 70)
        ctx.line_to(dx, 118)
        ctx.line_to(dx + 16, 118)
    ctx.stroke()
    rounded_rect(ctx, -58, -100, 116, 185, 58)
    ink(ctx, BODY[i], 8)
    for dx in (-22, 22):                                   # eyes
        if mood > 0.5:
            ctx.new_path()
            ctx.arc(dx, -38, 11, math.pi, TAU)
            ctx.set_source_rgba(*INK)
            ctx.set_line_width(6)
            ctx.stroke()
        else:
            ellipse(ctx, dx, -40, 8, 10)
            ctx.set_source_rgba(*INK)
            ctx.fill()
    ctx.new_path()                                         # mouth: smile, flat or frown
    if abs(mood) < 0.5:
        ctx.move_to(-14, -8)
        ctx.line_to(14, -8)
    elif mood > 0:
        ctx.arc(0, -16, 18, 0.15 * math.pi, 0.85 * math.pi)
    else:
        ctx.arc(0, 6, 18, 1.15 * math.pi, 1.85 * math.pi)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(6)
    ctx.stroke()
    if i == 0:                                             # hair tuft
        for a in (-0.5, 0.0, 0.5):
            ctx.move_to(0, -100)
            ctx.line_to(26 * math.sin(a), -100 - 30 * math.cos(a))
        ctx.set_line_width(7)
        ctx.stroke()
    elif i == 1:                                           # bandana
        ctx.new_path()
        ctx.arc(0, -42, 58, 1.12 * math.pi, 1.88 * math.pi)
        ctx.close_path()
        ink(ctx, YELLOW, 7)
    else:                                                  # glasses
        ctx.set_line_width(5)
        for dx in (-22, 22):
            ctx.new_path()
            ctx.arc(dx, -40, 19, 0, TAU)
            ctx.stroke()
        ctx.move_to(-3, -40)
        ctx.line_to(3, -40)
        ctx.stroke()
    ctx.restore()


def chicken(ctx, x, y, s, t, rooster=False, a=1.0, ph=0.0):
    if s <= 0.01 or a <= 0.01:
        return
    peck = max(0.0, math.sin(t * 2.6 + ph)) ** 6
    at(ctx, x, y, s)
    ctx.set_source_rgba(*INK, a)
    ctx.set_line_width(7)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for dx in (-14, 14):
        ctx.move_to(dx, 40)
        ctx.line_to(dx, 74)
        ctx.line_to(dx + 12, 74)
    ctx.stroke()
    body = hex_rgb("#C9772F") if rooster else WHITE
    ctx.new_path()                                         # tail
    ctx.move_to(-48, -6)
    ctx.line_to(-98 if rooster else -84, -62 if rooster else -40)
    ctx.line_to(-88, 8)
    ctx.close_path()
    ink(ctx, hex_rgb("#3B6B4A") if rooster else CREAM, 7, a)
    ellipse(ctx, 0, 0, 62, 50)
    ink(ctx, body, 7, a)
    hx, hy = 50 + 10 * peck, -52 + 46 * peck
    for k in range(3 if rooster else 2):                   # comb
        ellipse(ctx, hx - 12 + 13 * k, hy - 30, 9, 13 if rooster else 9)
        ink(ctx, RED, 5, a)
    ctx.new_path()
    ctx.move_to(hx + 24, hy - 6)
    ctx.line_to(hx + 50, hy + 4)
    ctx.line_to(hx + 24, hy + 12)
    ctx.close_path()
    ink(ctx, ORANGE, 5, a)
    ellipse(ctx, hx, hy, 30, 28)
    ink(ctx, body, 7, a)
    ellipse(ctx, hx + 10, hy - 4, 5, 6)
    ctx.set_source_rgba(*INK, a)
    ctx.fill()
    ctx.restore()


def chick(ctx, x, y, s, t, ph=0.0):
    if s <= 0.01:
        return
    at(ctx, x, y - 8 * abs(math.sin(t * 5 + ph)), s)
    ellipse(ctx, 0, 0, 34, 32)
    ink(ctx, YELLOW, 6)
    ctx.new_path()
    ctx.move_to(26, -4)
    ctx.line_to(46, 4)
    ctx.line_to(26, 10)
    ctx.close_path()
    ink(ctx, ORANGE, 4)
    ellipse(ctx, 12, -8, 4, 5)
    ctx.set_source_rgba(*INK)
    ctx.fill()
    ctx.restore()


def egg(ctx, x, y, s, rot=0.0):
    if s <= 0.01:
        return
    at(ctx, x, y, s, rot)
    ellipse(ctx, 0, 0, 30, 39)
    ink(ctx, CREAM, 6)
    ctx.restore()


def drumstick(ctx, x, y, s):
    if s <= 0.01:
        return
    at(ctx, x, y, s, -0.5)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(26)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.move_to(30, 0)
    ctx.line_to(92, 0)
    ctx.stroke()
    ctx.set_source_rgba(*CREAM)
    ctx.set_line_width(13)
    ctx.move_to(30, 0)
    ctx.line_to(92, 0)
    ctx.stroke()
    ellipse(ctx, -10, 0, 58, 44)
    ink(ctx, BROWN, 7)
    ctx.restore()


def bone(ctx, x, y, s, a=1.0):
    if s <= 0.01 or a <= 0.01:
        return
    at(ctx, x, y, s, -0.4)
    for lw, col in ((30, INK), (16, CREAM)):
        ctx.set_source_rgba(*col, a)
        ctx.set_line_width(lw)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(-50, 0)
        ctx.line_to(50, 0)
        ctx.stroke()
        for sx in (-56, 56):
            for sy in (-11, 11):
                ctx.new_path()
                ctx.arc(sx, sy, lw / 2 + 2, 0, TAU)
                ctx.fill()
    ctx.restore()


class Scene:
    W, H, FPS, nframes = W, H, FPS, NF

    def __init__(self, seed):
        self.seed = seed
        self.font = Font(FONT_PATH)
        self.bg = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        ctx = cairo.Context(self.bg)
        g = cairo.LinearGradient(0, 0, 0, H)
        g.add_color_stop_rgb(0, *SKY0)
        g.add_color_stop_rgb(1, *SKY1)
        ctx.set_source(g)
        ctx.paint()
        for i, y in enumerate(LTOP):
            rounded_rect(ctx, LX, y, LW, LH, 36)
            ink(ctx, PANEL[i], 8)
            ctx.set_source_rgba(*INK, 0.16)                # ground line
            ctx.rectangle(LX + 30, CY[i] + 92, LW - 60, 8)
            ctx.fill()

    def fit(self, text, width, size):
        return min(size, size * width / max(1.0, self.font.width(text, size)))

    def heading(self, ctx, t):
        k = max(j for j, h in enumerate(HEADS) if t >= h[0])
        t0, a, b = HEADS[k]
        s = 1.0 if k == 0 else lerp(0.6, 1.0, pop(t, t0, 0.3))
        sa = self.fit(a, 880, 118) * s
        self.font.draw(ctx, a, W / 2, 262, sa, WHITE, "c", stroke=INK, stroke_w=16, shadow=8)
        if b:
            self.font.draw(ctx, b, W / 2, 372, self.fit(b, 820, 72) * s, YELLOW, "c", stroke=INK, stroke_w=13)

    def label(self, ctx, i, t):
        cur = [l for l in LABELS[i] if t >= l[0]]
        if not cur:
            return
        t0, txt = cur[-1]
        big = t0 >= T_END
        size = (70 if big else 46) * lerp(0.5, 1.0, pop(t, t0, 0.3))
        w = self.font.width(txt, size)
        x, y = (LX + LW - 40 - w, LTOP[i] + 104) if big else (LX + 250, LTOP[i] + LH - 22)
        rounded_rect(ctx, x - 20, y - size * 0.86, w + 40, size * 1.08, 14)
        ink(ctx, INK if big else CREAM, 6)
        self.font.draw(ctx, txt, x, y, size, YELLOW if big else INK)

    def frame(self, i):
        t = i / FPS
        s = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        ctx = cairo.Context(s)
        ctx.set_source_surface(self.bg)
        ctx.paint()
        fade = 1.0 - ramp(t, T_RESET, T_RESET + 0.35)       # everything clears so the last frame cuts to the first
        mood = [0, 0, 0]
        if t >= 3.6:
            mood = [1 if t < T_WEEKS else (-1 if t < 18.6 else 0), 1 if t >= 4.2 else 0, 1 if t >= 5.5 else 0]
        if t >= T_RESET:
            mood = [0, 0, 0]
        for k in range(3):
            person(ctx, LX + 130, CY[k], k, mood[k], t)

        ctx.push_group()
        drop = lambda k: lerp(LTOP[k] - 90, CY[k] + 18, ease_out(ramp(t, 0.15 * k, 0.8 + 0.15 * k), 2)) \
            - 40 * math.sin(math.pi * ramp(t, 0.8 + 0.15 * k, 1.2 + 0.15 * k))

        def lane_clip(k):                                  # a falling chicken only shows inside its own lane
            ctx.save()
            ctx.rectangle(LX + 4, LTOP[k] + 4, LW - 8, LH - 8)
            ctx.clip()
        # lane A: eats it
        lane_clip(0)
        chicken(ctx, 480, drop(0), 1.0 - ramp(t, 3.4, 3.7), t)
        ctx.restore()
        drumstick(ctx, 500, CY[0] + 10, pop(t, 3.6) * (1.0 - ramp(t, T_WEEKS, T_WEEKS + 0.25)))
        bone(ctx, 500, CY[0] + 40, pop(t, 7.7), 1.0 - 0.65 * ramp(t, T_YEAR, T_YEAR + 0.6) - 0.35 * ramp(t, 17.8, 18.3))
        # lane B: keeps the hen, eats the eggs
        lane_clip(1)
        chicken(ctx, 480, drop(1), 1.0, t, ph=1.3)
        ctx.restore()
        for k, t0 in enumerate((4.2, 8.0, 8.4)):
            egg(ctx, 640 + 100 * k, CY[1] + 50, pop(t, t0))
        # lane C: breeds
        big = 1.0 - ramp(t, T_YEAR, T_YEAR + 0.3)
        lane_clip(2)
        chicken(ctx, 450, drop(2), big, t, ph=2.1)
        ctx.restore()
        chicken(ctx, lerp(1280, 690, ease_out(ramp(t, 4.8, 5.5))), CY[2] + 18, big, t, rooster=True, ph=0.4)
        for k, t0 in enumerate((8.2, 8.6, 9.0)):
            chick(ctx, 805 + 55 * k, CY[2] + 66, 0.62 * pop(t, t0) * big, t, k)
        for k, (fx, fy, t0) in enumerate(FLOCK):
            chicken(ctx, fx, CY[2] + fy + 14, 0.5 * pop(t, t0), t, rooster=(k % 5 == 1), ph=k * 0.9)
        # the hire: an egg flies from the flock to the first lane
        u = ramp(t, 17.8, 18.6)
        if u > 0:
            e = ease_in_out(u)
            egg(ctx, lerp(600, 500, e), lerp(CY[2], CY[0] + 30, e) - 150 * math.sin(math.pi * e),
                min(1.0, u * 6), TAU * e if u < 1 else 0.0)
        for k in range(3):
            self.label(ctx, k, t)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(fade)
        self.heading(ctx, t)
        s.flush()
        return bytes(s.get_data())

    def audio(self):
        """Placeholder mix: a light marimba-and-bass bed plus one cue per event. Measured only, never heard."""
        rng = np.random.default_rng(7)
        n = int(DUR * au.SR)
        buf = np.zeros((n + au.SR, 2))
        scale = (60, 64, 67, 69, 72, 67, 64, 62)           # C major pentatonic riff
        k = 0
        while k * 0.25 < DUR - 0.3:
            tt = k * 0.25
            au.place(buf, au.marimba(au.hz(scale[k % 8] + (0 if tt < T_YEAR else 5 if tt < T_HIRE else 0))), tt,
                     0.20, 0.3 * math.sin(k))
            if k % 4 == 0:
                au.place(buf, au.bass(au.hz(36 + (5 if T_YEAR <= tt < T_HIRE else 0)), 0.3), tt, 0.35)
            if k % 2 == 1:
                au.place(buf, au.shaker(rng), tt, 0.10)
            k += 1
        for j in range(3):                                 # chickens land
            au.place(buf, au.kick(), 0.8 + 0.15 * j, 0.5)
        pops = [3.6, 4.2, 5.5, 7.7, 8.0, 8.2, 8.4, 8.6, 9.0] + [f[2] for f in FLOCK]
        for j, tp in enumerate(pops):
            au.place(buf, au.pluck(au.hz(76 + (j * 3) % 10), 0.4), tp, 0.30)
        for h in HEADS[1:]:
            au.place(buf, au.clap(rng), h[0], 0.30)
        au.place(buf, au.glide(330, 150, 0.6), 7.7, 0.30)   # nothing left
        au.place(buf, au.glide(400, 900, 0.7), 17.8, 0.25)  # the egg flies
        for j, m in enumerate((72, 76, 79)):
            au.place(buf, au.bell(au.hz(m)), T_END + 0.2 + 0.3 * j, 0.35)
        buf[n - int(0.1 * au.SR):n] *= np.linspace(1, 0, int(0.1 * au.SR))[:, None]
        return buf[:n]

    def metadata(self):
        return dict(
            title="Everyone gets 1 chicken. Who ends up rich?",
            description=("Three people start with the same chicken. One eats it, one keeps it for eggs, one breeds "
                         "it. A year later they live very differently. A short parable about spending, saving and "
                         "investing, drawn by code with synthesized sound: no footage, samples or stock media."),
            tags=["parable", "money", "saving", "investing", "compound growth", "animation", "made with code"],
            made_for_kids=False,
            synthetic_media_disclosure=False,
            seed=self.seed,
            drama=dict(note="story format, no simulation"),
        )


def score_seed(seed):
    return dict(seed=seed, score=0)
